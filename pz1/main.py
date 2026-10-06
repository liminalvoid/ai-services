import logging
import time
from typing import Literal, Optional

import model
import storage
from fastapi import FastAPI, HTTPException, Query, Request, status
from schemas import FarmRequest, HealthResponse, ModelInfoResponse, PredictionResponse
from services import UnknownRegionError, make_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


app = FastAPI(
    title="Agro Scoring API",
    description=(
        "REST API для оценки риска "
        "сельскохозяйственных предприятий."
    ),
    version="1.0.0"
)


@app.middleware("http")
async def add_process_time(
    request: Request,
    call_next
):
    """
    Добавляет в HTTP-ответ заголовок X-Process-Time,
    содержащий время обработки запроса.
    """

    start_time = time.perf_counter()

    response = await call_next(request)

    process_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = str(
        round(process_time, 6)
    )

    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
    description=(
        "Используется для проверки того, "
        "что REST API запущен и отвечает."
    )
)
def health():
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
def model_info():
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
    status_code=status.HTTP_201_CREATED,
    summary="Оценить риск хозяйства",
    description=(
        "Принимает характеристики хозяйства, "
        "выполняет валидацию, инференс модели, "
        "сохраняет и возвращает оценку риска."
    ),
    responses={
        400: {"description": "Регион не входит в справочник"},
        422: {"description": "Ошибка валидации Pydantic"},
        503: {"description": "Модель недоступна"}
    }
)
def predict(request: FarmRequest):

    if not model.MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is temporarily unavailable"
        )

    try:
        return make_prediction(request)

    except UnknownRegionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        )


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    summary="Получить список прогнозов",
    description=(
        "Возвращает последние выполненные прогнозы "
        "(новые первыми). Поддерживает ограничение "
        "количества результатов и фильтрацию по уровню риска."
    ),
    responses={
        422: {
            "description": (
                "limit вне диапазона 1..100 "
                "или недопустимый risk_level"
            )
        }
    }
)
def get_predictions(

    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Максимальное количество результатов"
    ),

    # Literal: Pydantic сам отклонит другие значения (422),
    # а Swagger покажет выпадающий список
    risk_level: Optional[Literal["low", "medium", "high"]] = Query(
        default=None,
        description=(
            "Фильтр по категории риска: "
            "low, medium или high"
        )
    )

):

    return storage.list_predictions(limit, risk_level)


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получить прогноз по request_id",
    description=(
        "Возвращает сохраненный прогноз "
        "по его уникальному идентификатору."
    ),
    responses={
        404: {"description": "Прогноз не найден"}
    }
)
def get_prediction(request_id: str):

    result = storage.get_prediction(request_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
