"""
Доменные ошибки сервиса.

Модули бизнес-логики, хранилища, очереди и LLM-клиента
не знают об HTTP. Каждая ошибка несет свой HTTP-код,
а main.py превращает ее в JSON-ответ с request_id.
"""

from fastapi import status


class ServiceError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    headers: dict | None = None

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class BusinessRuleError(ServiceError):
    """Нарушено бизнес-правило (например, регион вне справочника)."""
    status_code = status.HTTP_400_BAD_REQUEST


class NotFoundError(ServiceError):
    status_code = status.HTTP_404_NOT_FOUND


class AmbiguousResultError(ServiceError):
    """request_id относится к batch и содержит несколько прогнозов."""
    status_code = status.HTTP_409_CONFLICT


class QueueFullError(ServiceError):
    """Очередь инференса переполнена (backpressure)."""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    headers = {"Retry-After": "1"}


class LLMBadResponseError(ServiceError):
    """Внешний LLM API вернул ошибку или некорректный ответ."""
    status_code = status.HTTP_502_BAD_GATEWAY


class ServiceUnavailableError(ServiceError):
    """Модель, БД или внешний сервис недоступны."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class ServiceTimeoutError(ServiceError):
    """Превышен таймаут инференса или LLM."""
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
