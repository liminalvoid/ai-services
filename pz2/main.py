import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import batching
import config
import llm
import model
import services
import storage
from errors import AmbiguousResultError, BusinessRuleError, NotFoundError, ServiceError
from logs import request_id_ctx, setup_logging
from schemas import (
    BatchPredictionResponse,
    BatchRequest,
    ErrorResponse,
    FarmRequest,
    HealthResponse,
    LLMChatRequest,
    LLMChatResponse,
    ModelInfoResponse,
    PaginatedPredictionsResponse,
    PredictionResponse,
    PredictWithLLMResponse,
    QueueStatusResponse,
    SeedResponse,
    StoredPredictionResponse,
)

setup_logging()

logger = logging.getLogger(__name__)


ERROR_DESCRIPTIONS = {
    400: "Нарушение бизнес-правил",
    404: "Результат не найден",
    409: "request_id относится к batch и содержит несколько прогнозов",
    422: "Ошибка валидации Pydantic",
    429: "Очередь инференса переполнена",
    502: "Ошибка внешнего LLM-провайдера",
    503: "Модель, БД или внешний сервис недоступны",
    504: "Превышен таймаут инференса или LLM",
}


def error_responses(*codes: int) -> dict:
    """
    Описание возможных ошибок для OpenAPI / Swagger.
    """

    return {
        code: {"model": ErrorResponse, "description": ERROR_DESCRIPTIONS[code]}
        for code in codes
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Старт: инициализация SQLite, HTTP-клиента LLM и batch worker.
    Остановка: корректное завершение worker и закрытие клиента.
    """

    await storage.init_db()
    app.state.llm_client = llm.create_client()
    worker = batching.start_worker()

    yield

    await batching.stop_worker(worker)
    await app.state.llm_client.aclose()
    logger.info("Application stopped")


app = FastAPI(
    title="Agro Scoring API",
    description=(
        "REST API для оценки риска "
        "сельскохозяйственных предприятий: request_id, SQLite (aiosqlite), "
        "explicit и dynamic batching, backpressure, интеграция с LLM."
    ),
    version="2.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next
):
    """
    1. Берет X-Request-ID клиента или генерирует UUID.
    2. Сохраняет его в request.state и в контекст логирования.
    3. Измеряет время обработки.
    4. Возвращает X-Request-ID и X-Process-Time в заголовках.
    """

    start_time = time.perf_counter()

    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    token = request_id_ctx.set(request_id)

    logger.info(
        "Request started | method=%s | path=%s",
        request.method,
        request.url.path
    )

    try:
        response = await call_next(request)

    except Exception:
        logger.exception("Unhandled error")
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id}
        )

    process_time = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(round(process_time, 6))

    logger.info(
        "Request completed | status=%s | time=%.4f",
        response.status_code,
        process_time
    )

    request_id_ctx.reset(token)

    return response


def error_response(
    request: Request,
    status_code: int,
    detail,
    headers: dict | None = None
) -> JSONResponse:
    """
    Единый формат ошибки: detail + request_id
    (если запрос уже прошел через middleware).
    """

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None)
        },
        headers=headers
    )


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, error: ServiceError):
    return error_response(request, error.status_code, error.detail, error.headers)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, error: StarletteHTTPException):
    return error_response(request, error.status_code, error.detail, error.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, error: RequestValidationError):
    return error_response(
        request,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        jsonable_encoder(error.errors())
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
    description=(
        "Используется для проверки того, "
        "что REST API запущен и отвечает."
    )
)
async def health():
    return {
        "status": "ok"
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
    description=(
        "Возвращает название, версию, тип "
        "и текущее состояние модели."
    )
)
async def model_info():
    return {
        "model_name": model.MODEL_NAME,
        "model_version": model.MODEL_VERSION,
        "model_type": model.MODEL_TYPE,
        "status": (
            "ready"
            if model.MODEL_READY
            else "unavailable"
        )
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Оценить риск хозяйства",
    description=(
        "Одиночный инференс: валидация, вызов модели, "
        "сохранение в SQLite. 200 OK возвращается "
        "только после успешного COMMIT."
    ),
    responses=error_responses(400, 422, 503, 504)
)
async def predict(data: FarmRequest, request: Request):
    result = await services.predict_single(data, request.state.request_id)
    await storage.save_prediction(result, source="single")
    return result


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    summary="Явный пакетный инференс",
    description=(
        f"Принимает до {config.MAX_BATCH_SIZE} объектов, обрабатывает их "
        "одним вызовом модели и сохраняет одной транзакцией "
        "(executemany + commit). Все прогнозы получают общий request_id."
    ),
    responses=error_responses(400, 422, 503, 504)
)
async def batch_predict(batch: BatchRequest, request: Request):
    request_id = request.state.request_id
    start_time = time.perf_counter()

    items = await services.predict_batch(batch.items)
    await storage.save_predictions_batch(request_id, items)

    return {
        "request_id": request_id,
        "batch_size": len(items),
        "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "items": items
    }


@app.post(
    "/predict-queued",
    response_model=PredictionResponse,
    summary="Инференс через очередь (dynamic batching)",
    description=(
        "Клиент отправляет один объект, сервер ставит его в очередь, "
        "а worker объединяет запросы в batch "
        f"(до {config.MAX_BATCH_SIZE} объектов или {config.MAX_BATCH_WAIT_MS} мс). "
        "При переполнении очереди — 429."
    ),
    responses=error_responses(400, 422, 429, 503, 504)
)
async def predict_queued(data: FarmRequest, request: Request):
    request_id = request.state.request_id

    services.ensure_model_ready()
    services.validate_business_rules(data)

    future = batching.enqueue(data, request_id)

    # При таймауте wait_for отменяет Future, и worker пропустит задачу
    score = await services.with_inference_timeout(future)

    result = services.make_prediction_result(data, score, request_id)
    await storage.save_prediction(result, source="queued")
    return result


@app.get(
    "/queue-status",
    response_model=QueueStatusResponse,
    summary="Состояние очереди и параметры batching"
)
async def queue_status():
    return {
        "queue_size": batching.queue_size(),
        "queue_max_size": config.QUEUE_MAX_SIZE,
        "max_batch_size": config.MAX_BATCH_SIZE,
        "max_batch_wait_ms": config.MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": config.INFERENCE_TIMEOUT_SECONDS
    }


@app.get(
    "/predictions",
    response_model=list[StoredPredictionResponse],
    summary="Получить историю прогнозов из SQLite",
    description=(
        "Возвращает последние выполненные прогнозы "
        "(новые первыми). Поддерживает ограничение "
        "количества результатов и фильтрацию по уровню риска."
    ),
    responses=error_responses(422, 503)
)
async def get_predictions(

    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Максимальное количество результатов"
    ),

    # Literal: Pydantic сам отклонит другие значения (422),
    # а Swagger покажет выпадающий список
    risk_level: Literal["low", "medium", "high"] | None = Query(
        default=None,
        description=(
            "Фильтр по категории риска: "
            "low, medium или high"
        )
    )

):

    return await storage.list_predictions(limit, risk_level)


@app.get(
    "/predictions/by-weekday",
    response_model=PaginatedPredictionsResponse,
    summary="Прогнозы по дню недели с пагинацией",
    description=(
        "Offset-пагинация (LIMIT/OFFSET + COUNT(*)). Пример: "
        "/predictions/by-weekday?weekday=monday&page=1&per_page=5"
    ),
    responses=error_responses(400, 404, 422, 503)
)
async def get_predictions_by_weekday(
    request: Request,

    weekday: str = Query(
        ...,
        description=(
            "День недели: monday, tuesday, wednesday, "
            "thursday, friday, saturday, sunday"
        ),
        examples=["monday"]
    ),

    page: int = Query(
        default=1,
        ge=1,
        description="Номер страницы"
    ),

    per_page: int = Query(
        default=5,
        ge=1,
        le=100,
        description="Количество записей на странице"
    )

):
    weekday = weekday.lower()

    # Не Literal, а проверка вручную: неизвестный день недели —
    # это ошибка предметной области (400), а не формата (422)
    if weekday not in storage.WEEKDAY_MAP:
        raise BusinessRuleError(
            f"Unknown weekday: {weekday}. "
            f"Allowed: {', '.join(storage.WEEKDAY_MAP)}"
        )

    data, total_items, total_pages = await storage.get_predictions_by_weekday_page(
        weekday,
        page,
        per_page
    )

    if page > max(total_pages, 1):
        raise NotFoundError(
            f"Page {page} does not exist. Total pages: {total_pages}"
        )

    def page_url(number: int) -> str:
        return str(request.url.include_query_params(
            weekday=weekday,
            page=number,
            per_page=per_page
        ))

    return {
        "data": data,
        "meta": {
            "weekday": weekday,
            "current_page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "next_page": page_url(page + 1) if page < total_pages else None,
            "prev_page": page_url(page - 1) if page > 1 else None
        }
    }


@app.post(
    "/demo/seed-weekdays",
    response_model=SeedResponse,
    summary="Создать демонстрационные прогнозы за разные дни",
    description=(
        "Добавляет записи с датами за последние 21 день, "
        "чтобы проверить выборку по дню недели и пагинацию."
    ),
    responses=error_responses(422, 503)
)
async def seed_demo_weekdays(
    count: int = Query(
        default=70,
        ge=7,
        le=1000,
        description="Количество демонстрационных записей"
    )
):
    await storage.save_seed_predictions(services.make_demo_predictions(count))
    return {"status": "ok", "created": count}


@app.get(
    "/predictions/{request_id}",
    response_model=StoredPredictionResponse,
    summary="Получить прогноз по request_id",
    description=(
        "Возвращает одиночный прогноз. Если request_id "
        "относится к batch, возвращается 409 — используйте "
        "/requests/{request_id}/predictions."
    ),
    responses=error_responses(404, 409, 503)
)
async def get_prediction(request_id: str):
    results = await storage.get_predictions_by_request_id(request_id)

    if not results:
        raise NotFoundError("Prediction not found")

    if len(results) > 1:
        raise AmbiguousResultError(
            "This request_id contains multiple predictions. "
            f"Use /requests/{request_id}/predictions"
        )

    return results[0]


@app.get(
    "/requests/{request_id}/predictions",
    response_model=list[StoredPredictionResponse],
    summary="Все прогнозы одного HTTP-запроса",
    description=(
        "Для /predict и /predict-queued — один элемент, "
        "для /batch-predict — все элементы пакета."
    ),
    responses=error_responses(404, 503)
)
async def get_request_predictions(request_id: str):
    results = await storage.get_predictions_by_request_id(request_id)

    if not results:
        raise NotFoundError("Prediction not found")

    return results


@app.post(
    "/llm/chat",
    response_model=LLMChatResponse,
    summary="Прямой запрос к LLM (Vireonix)",
    description=(
        "Отправляет prompt в OpenAI-совместимый Vireonix API "
        "через httpx.AsyncClient. Не передавайте реальные "
        "конфиденциальные данные."
    ),
    responses=error_responses(422, 502, 503, 504)
)
async def llm_chat(data: LLMChatRequest, request: Request):
    return await llm.chat(
        request.app.state.llm_client,
        prompt=data.prompt,
        request_id=request.state.request_id,
        model=data.model,
        temperature=data.temperature,
        max_tokens=data.max_tokens
    )


@app.post(
    "/predict-with-llm",
    response_model=PredictWithLLMResponse,
    summary="Агроскоринг с текстовым объяснением от LLM",
    description=(
        "Сначала детерминированная модель считает risk_score и "
        "risk_level, результат сохраняется в SQLite. Затем LLM "
        "формирует только текстовое объяснение — оценку она не меняет."
    ),
    responses=error_responses(400, 422, 502, 503, 504)
)
async def predict_with_llm(data: FarmRequest, request: Request):
    request_id = request.state.request_id

    prediction = await services.predict_single(data, request_id)
    await storage.save_prediction(prediction, source="single+llm")

    explanation = await llm.chat(
        request.app.state.llm_client,
        prompt=llm.build_explanation_prompt(prediction),
        request_id=request_id
    )

    return {
        "prediction": prediction,
        "llm": explanation
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
