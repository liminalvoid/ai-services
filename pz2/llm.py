"""
Клиент внешней LLM (Vireonix, OpenAI-совместимый формат
POST /v1/chat/completions) на httpx.AsyncClient.
"""

import logging
import time

import httpx

import config
from errors import LLMBadResponseError, ServiceTimeoutError, ServiceUnavailableError


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты помощник сервиса агроскоринга. "
    "Отвечай кратко и понятно. "
    "Не изменяй переданный risk_score и risk_level. "
    "Не придумывай причины, которых нет во входных данных."
)


def create_client() -> httpx.AsyncClient:
    """
    Один клиент на все приложение: создается в lifespan
    при старте и закрывается при остановке.
    """

    return httpx.AsyncClient(
        timeout=httpx.Timeout(config.LLM_TIMEOUT_SECONDS)
    )


async def chat(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: str,
    model: str = config.LLM_MODEL,
    temperature: float = config.LLM_DEFAULT_TEMPERATURE,
    max_tokens: int = config.LLM_DEFAULT_MAX_TOKENS
) -> dict:
    """
    Отправляет prompt в LLM. Ошибки внешнего сервиса:
    таймаут -> 504, нет соединения -> 503,
    ошибка или некорректный ответ провайдера -> 502.
    """

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    logger.info("LLM request | model=%s | prompt_chars=%s", model, len(prompt))

    started = time.perf_counter()

    try:
        response = await client.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            json=payload,
            # Передаем request_id дальше — сквозная идентификация
            headers={"X-Request-ID": request_id}
        )

    except httpx.TimeoutException as error:
        logger.error("LLM timeout")
        raise ServiceTimeoutError("LLM request timeout") from error

    except httpx.RequestError as error:
        logger.error("LLM connection error | %r", error)
        raise ServiceUnavailableError("LLM service is unavailable") from error

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if response.status_code != 200:
        logger.error(
            "LLM upstream error | status=%s | body=%s",
            response.status_code,
            response.text[:300]
        )
        raise LLMBadResponseError(
            f"LLM provider returned an error (HTTP {response.status_code})"
        )

    try:
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LLMBadResponseError("LLM provider returned an invalid response") from error

    if not content:
        raise LLMBadResponseError("LLM response does not contain message content")

    logger.info(
        "LLM response | returned_model=%s | finish_reason=%s | latency_ms=%s",
        data.get("model"),
        choice.get("finish_reason"),
        latency_ms
    )

    return {
        "request_id": request_id,
        "requested_model": model,
        "returned_model": data.get("model"),
        "answer": content.strip(),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "latency_ms": latency_ms
    }


def build_explanation_prompt(prediction: dict) -> str:
    """
    Промпт строится только из результата скоринга: farm_id
    и исходные финансовые признаки во внешний сервис не уходят.
    """

    return (
        "Сформируй краткое объяснение для оператора банка "
        "по результату алгоритмического агроскоринга.\n\n"
        f"risk_score: {prediction['risk_score']}\n"
        f"risk_level: {prediction['risk_level']}\n"
        f"базовая рекомендация: {prediction['recommendation']}\n"
        f"версия скоринговой модели: {prediction['model_version']}\n\n"
        "Требования к ответу:\n"
        "1. 2-4 предложения.\n"
        "2. Не меняй категорию риска.\n"
        "3. Не придумывай дополнительные факторы риска.\n"
        "4. Укажи, что решение требует применения правил банка "
        "и при необходимости проверки оператором."
    )
