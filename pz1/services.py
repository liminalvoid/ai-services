import uuid
import logging

from schemas import FarmRequest
from model import calculate_risk, MODEL_VERSION
import storage


logger = logging.getLogger(__name__)


# Разрешенные регионы.
# Это пример бизнес-справочника.
ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol"
}


class UnknownRegionError(ValueError):
    """
    Нарушено бизнес-правило: регион не входит в справочник.
    """


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


def validate_business_rules(request: FarmRequest) -> None:
    """
    Pydantic уже проверил типы, обязательность и диапазоны.
    Здесь проверяются правила, которые зависят от справочников.
    """

    if request.region not in ALLOWED_REGIONS:
        raise UnknownRegionError(
            f"Unknown region: {request.region}. "
            f"Allowed regions: {sorted(ALLOWED_REGIONS)}"
        )


def make_prediction(request: FarmRequest) -> dict:
    validate_business_rules(request)

    logger.info(
        "Prediction request received | farm_id=%s",
        request.farm_id
    )

    score = calculate_risk(request)

    level = get_risk_level(score)

    request_id = str(uuid.uuid4())

    result = {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": MODEL_VERSION
    }

    storage.save_prediction(result)

    logger.info(
        "Prediction completed | "
        "request_id=%s | "
        "farm_id=%s | "
        "risk_score=%s | "
        "risk_level=%s",
        request_id,
        request.farm_id,
        score,
        level
    )

    return result
