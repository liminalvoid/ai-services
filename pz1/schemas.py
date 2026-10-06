from pydantic import BaseModel, Field


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


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str
