import asyncio

import config
from schemas import FarmRequest

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"

# Имитируем состояние модели.
# Если установить False, endpoints инференса возвращают HTTP 503.
MODEL_READY = True

# Один lock имитирует один экземпляр модели:
# вызовы выполняются строго по одному.
model_lock = asyncio.Lock()


def calculate_risk(data: FarmRequest) -> float:
    """
    Реализует простую логику оценки риска по финансовым показателям
    и отраслевым факторам.
    """

    score = 0.1

    # Длительная просрочка увеличивает риск
    if data.payment_delay_days > 30:
        score += 0.3

    # Наличие предыдущих дефолтов увеличивает риск
    if data.previous_defaults > 0:
        score += 0.3

    # Большая задолженность увеличивает риск
    if data.debt > 5_000_000:
        score += 0.2

    # Малое количество осадков условно увеличивает
    # аграрный риск
    if data.precipitation_mm < 100:
        score += 0.1

    # Ограничиваем значение диапазоном от 0 до 1
    score = min(score, 1.0)

    return round(score, 2)


async def model_predict_single(data: FarmRequest) -> float:
    """
    Одиночный вызов модели: фиксированный overhead
    + время на один объект.
    """

    async with model_lock:
        await asyncio.sleep(
            config.MODEL_FIXED_OVERHEAD_SECONDS
            + config.MODEL_PER_ITEM_SECONDS
        )
        return calculate_risk(data)


async def model_predict_batch(items: list[FarmRequest]) -> list[float]:
    """
    Один пакетный вызов модели: overhead платится
    один раз на весь batch.
    """

    async with model_lock:
        await asyncio.sleep(
            config.MODEL_FIXED_OVERHEAD_SECONDS
            + config.MODEL_PER_ITEM_SECONDS * len(items)
        )
        return [calculate_risk(item) for item in items]
