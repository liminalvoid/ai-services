from pydantic import BaseModel, Field

import config


class FarmRequest(BaseModel):
    """
    Данные сельскохозяйственного предприятия,
    которые клиент передает в POST /predict
    """

    farm_id: str = Field(
        ...,
        min_length=1,
        description="Идентификатор хозяйства",
        examples=["FARM-001"]
    )

    region: str = Field(
        ...,
        min_length=1,
        description="Регион хозяйства",
        examples=["Krasnodar"]
    )

    crop_type: str = Field(
        ...,
        min_length=1,
        description="Основная сельскохозяйственная культура",
        examples=["wheat"]
    )

    area_ha: float = Field(
        ...,
        gt=0,
        description="Площадь посевов, га",
        examples=[2500]
    )

    temperature_avg: float = Field(
        ...,
        ge=-60,
        le=60,
        description="Средняя температура, °C",
        examples=[24.3]
    )

    precipitation_mm: float = Field(
        ...,
        ge=0,
        description="Количество осадков, мм",
        examples=[320]
    )

    payment_delay_days: int = Field(
        ...,
        ge=0,
        description="Количество дней просрочки платежа",
        examples=[15]
    )

    previous_defaults: int = Field(
        ...,
        ge=0,
        description="Количество предыдущих дефолтов",
        examples=[0]
    )

    debt: float = Field(
        ...,
        ge=0,
        description="Текущая задолженность",
        examples=[1500000]
    )


class PredictionResponse(BaseModel):
    """
    Структура ответа сервиса после выполнения прогноза.
    """

    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class StoredPredictionResponse(PredictionResponse):
    """
    Прогноз, прочитанный из SQLite: дополнительно
    содержит источник записи и время создания.
    """

    source: str = Field(
        description="single, queued, batch, single+llm или seed"
    )
    created_at: str = Field(
        description="Время записи (UTC)"
    )


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


class ErrorResponse(BaseModel):
    detail: str | list
    request_id: str | None


# --- Explicit batch ---

class BatchRequest(BaseModel):
    """
    Явный batch: клиент передает несколько хозяйств
    одним HTTP-запросом.
    """

    items: list[FarmRequest] = Field(
        ...,
        min_length=1,
        max_length=config.MAX_BATCH_SIZE,
        description=(
            "Список объектов для пакетного инференса "
            f"(не более {config.MAX_BATCH_SIZE})"
        )
    )


class BatchItemResponse(BaseModel):
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: list[BatchItemResponse]


# --- Очередь ---

class QueueStatusResponse(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: int
    inference_timeout_seconds: float


# --- Пагинация ---

class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: str | None
    prev_page: str | None


class PaginatedPredictionsResponse(BaseModel):
    data: list[StoredPredictionResponse]
    meta: PaginationMeta


class SeedResponse(BaseModel):
    status: str
    created: int


# --- LLM ---

class LLMChatRequest(BaseModel):
    """
    Запрос к внешней LLM. Не передавайте сюда реальные
    конфиденциальные банковские или персональные данные.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Текст запроса к LLM",
        examples=["Объясни простыми словами, что такое агроскоринг"]
    )

    model: str = Field(
        default=config.LLM_MODEL,
        min_length=1,
        description="Модель Vireonix"
    )

    temperature: float = Field(
        default=config.LLM_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0
    )

    max_tokens: int = Field(
        default=config.LLM_DEFAULT_MAX_TOKENS,
        ge=1,
        le=2000
    )


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None
    answer: str
    finish_reason: str | None = None
    usage: dict | None = None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    """
    Детерминированный результат скоринга
    и отдельно — текстовое объяснение от LLM.
    """

    prediction: PredictionResponse
    llm: LLMChatResponse
