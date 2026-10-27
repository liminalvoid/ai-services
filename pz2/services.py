import asyncio
import logging
import uuid

import config
import model
from errors import BusinessRuleError, ServiceTimeoutError, ServiceUnavailableError
from schemas import FarmRequest


logger = logging.getLogger(__name__)


# Разрешенные регионы.
# Это пример бизнес-справочника.
ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol"
}


def get_risk_level(score: float) -> str:
    """
    Преобразует score в категорию риска.
    """

    if score < 0.3:
        return "low"

    if score < 0.7:
        return "medium"

    return "high"


def get_recommendation(level: str) -> str:
    """
    Возвращает рекомендацию в зависимости от категории риска.
    """

    if level == "low":
        return "Стандартное рассмотрение"

    if level == "medium":
        return "Требуется дополнительная проверка"

    return "Высокий риск. Требуется ручное рассмотрение"


def ensure_model_ready() -> None:
    if not model.MODEL_READY:
        raise ServiceUnavailableError("Model is temporarily unavailable")


def validate_business_rules(request: FarmRequest) -> None:
    """
    Pydantic уже проверил типы, обязательность и диапазоны.
    Здесь проверяются правила, которые зависят от справочников.
    """

    if request.region not in ALLOWED_REGIONS:
        raise BusinessRuleError(
            f"Unknown region: {request.region}. "
            f"Allowed regions: {sorted(ALLOWED_REGIONS)}"
        )


def make_prediction_result(
    request: FarmRequest,
    score: float,
    request_id: str
) -> dict:
    """
    Постпроцессинг: score -> категория риска и рекомендация.
    """

    level = get_risk_level(score)

    return {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": model.MODEL_VERSION
    }


async def with_inference_timeout(awaitable, detail: str = "Inference timeout"):
    """
    Ограничивает время инференса INFERENCE_TIMEOUT_SECONDS,
    при превышении — 504.
    """

    try:
        return await asyncio.wait_for(
            awaitable,
            timeout=config.INFERENCE_TIMEOUT_SECONDS
        )

    except TimeoutError as error:
        logger.warning(
            "%s | timeout=%ss",
            detail,
            config.INFERENCE_TIMEOUT_SECONDS
        )
        raise ServiceTimeoutError(detail) from error


async def predict_single(request: FarmRequest, request_id: str) -> dict:
    """
    Контур одиночного инференса: проверки -> модель -> постпроцессинг.
    """

    ensure_model_ready()
    validate_business_rules(request)

    logger.info(
        "Prediction request received | farm_id=%s",
        request.farm_id
    )

    score = await with_inference_timeout(
        model.model_predict_single(request)
    )

    result = make_prediction_result(request, score, request_id)

    logger.info(
        "Prediction completed | "
        "farm_id=%s | "
        "risk_score=%s | "
        "risk_level=%s",
        request.farm_id,
        score,
        result["risk_level"]
    )

    return result


async def predict_batch(items: list[FarmRequest]) -> list[dict]:
    """
    Explicit batch: все объекты обрабатываются
    одним вызовом model_predict_batch().
    """

    ensure_model_ready()

    for item in items:
        validate_business_rules(item)

    scores = await with_inference_timeout(
        model.model_predict_batch(items),
        detail="Batch inference timeout"
    )

    results = []

    for item, score in zip(items, scores):
        level = get_risk_level(score)
        results.append({
            "farm_id": item.farm_id,
            "risk_score": score,
            "risk_level": level,
            "recommendation": get_recommendation(level),
            "model_version": model.MODEL_VERSION
        })

    logger.info(
        "Batch inference completed | batch_size=%s",
        len(items)
    )

    return results


def make_demo_predictions(count: int) -> list[dict]:
    """
    Демонстрационные прогнозы, распределенные по последним
    21 дню: по 3 недели на каждый день недели.
    """

    items = []

    for i in range(count):
        score = round(min(0.1 + (i % 9) * 0.1, 1.0), 2)
        level = get_risk_level(score)
        items.append({
            "request_id": str(uuid.uuid4()),
            "farm_id": f"DEMO-{i + 1:03d}",
            "risk_score": score,
            "risk_level": level,
            "recommendation": get_recommendation(level),
            "model_version": model.MODEL_VERSION,
            "days_ago": i % 21
        })

    return items
