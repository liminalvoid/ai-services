"""
Тест-кейсы лабораторной работы №2 (задание 15).

Приложение запускается в процессе вместе с lifespan
(SQLite, batch worker, HTTP-клиент LLM), запросы идут через
httpx.AsyncClient + ASGITransport, поэтому можно отправлять
их конкурентно и проверить очередь, 429 и dynamic batching.

Используется отдельная БД test_agro_scoring.db.
Кейсы с LLM обращаются к реальному Vireonix API.
"""

import asyncio
import time

import httpx

import config
import model

config.DATABASE_PATH = config.BASE_DIR / "test_agro_scoring.db"

from main import app  # noqa: E402

VALID = {
    "farm_id": "FARM-001",
    "region": "Krasnodar",
    "crop_type": "wheat",
    "area_ha": 2500,
    "temperature_avg": 24.3,
    "precipitation_mm": 320,
    "payment_delay_days": 45,
    "previous_defaults": 1,
    "debt": 6500000
}


def farm(i: int) -> dict:
    return {**VALID, "farm_id": f"FARM-{i:03d}"}


async def flood_queue(client: httpx.AsyncClient, n: int) -> list[httpx.Response]:
    """
    n одновременных запросов к /predict-queued:
    при n > QUEUE_MAX_SIZE часть должна получить 429.
    """

    return await asyncio.gather(*[
        client.post("/predict-queued", json=farm(i))
        for i in range(n)
    ])


async def main():
    for suffix in ("", "-wal", "-shm"):
        config.DATABASE_PATH.with_name(config.DATABASE_PATH.name + suffix).unlink(missing_ok=True)

    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=120
    ) as client:

        created = await client.post("/predict", json=VALID, headers={"X-Request-ID": "lab2-demo-request"})
        batch = await client.post("/batch-predict", json={"items": [farm(i) for i in range(1, 4)]})
        batch_id = batch.json()["request_id"]
        await client.post("/demo/seed-weekdays", params={"count": 70})

        flood = await flood_queue(client, 300)
        flood_codes = [r.status_code for r in flood]

        llm_ok = await client.post("/llm/chat", json={
            "prompt": "Одним предложением: зачем банку агроскоринг?",
            "max_tokens": 100
        })
        with_llm = await client.post("/predict-with-llm", json=VALID)

        # Недоступный LLM-провайдер: соединение отклонено -> 503
        real_url = config.LLM_BASE_URL
        config.LLM_BASE_URL = "http://127.0.0.1:9/v1"
        llm_down = await client.post("/llm/chat", json={"prompt": "test"})
        config.LLM_BASE_URL = real_url

        # Таймаут инференса (задание 11)
        config.INFERENCE_TIMEOUT_SECONDS = 0.001
        timeout_single = await client.post("/predict", json=VALID)
        timeout_queued = await client.post("/predict-queued", json=VALID)
        config.INFERENCE_TIMEOUT_SECONDS = 2.0

        # Модель недоступна
        model.MODEL_READY = False
        model_down = await client.post("/predict", json=VALID)
        model.MODEL_READY = True

        # Необработанная ошибка внутри модели -> 500 с request_id
        original = model.calculate_risk
        model.calculate_risk = lambda data: 1 / 0
        internal = await client.post("/predict", json=VALID)
        model.calculate_risk = original

        # БД недоступна
        db_path = config.DATABASE_PATH
        config.DATABASE_PATH = config.BASE_DIR / "missing-dir" / "x.db"
        db_down = await client.get("/predictions")
        config.DATABASE_PATH = db_path

        cases = [
            ("GET /health", 200, await client.get("/health")),
            ("корректный POST /predict", 200, created),
            ("area_ha = -100", 422, await client.post("/predict", json={**VALID, "area_ha": -100})),
            ("неизвестный регион", 400, await client.post("/predict", json={**VALID, "region": "Moscow"})),
            ("существующий request_id", 200, await client.get("/predictions/lab2-demo-request")),
            ("неизвестный request_id", 404, await client.get("/predictions/abc-123")),
            ("корректный POST /batch-predict", 200, batch),
            ("batch > MAX_BATCH_SIZE", 422, await client.post("/batch-predict", json={"items": [farm(i) for i in range(9)]})),
            ("POST /predict-queued", 200, await client.post("/predict-queued", json=VALID)),
            ("переполненная очередь", 429, next((r for r in flood if r.status_code == 429), flood[0])),
            ("by-weekday page=1 per_page=5", 200, await client.get("/predictions/by-weekday", params={"weekday": "monday", "page": 1, "per_page": 5})),
            ("page = -1", 422, await client.get("/predictions/by-weekday", params={"weekday": "monday", "page": -1})),
            ("неизвестный weekday", 400, await client.get("/predictions/by-weekday", params={"weekday": "funday"})),
            ("корректный POST /llm/chat", 200, llm_ok),
            ("недоступный LLM", 503, llm_down),
            ("корректный /predict-with-llm", 200, with_llm),
            # Дополнительные проверки
            ("batch request_id в /predictions/{id}", 409, await client.get(f"/predictions/{batch_id}")),
            ("GET /requests/{batch_id}/predictions", 200, await client.get(f"/requests/{batch_id}/predictions")),
            ("timeout 0.001 с, /predict", 504, timeout_single),
            ("timeout 0.001 с, /predict-queued", 504, timeout_queued),
            ("MODEL_READY = False", 503, model_down),
            ("БД недоступна", 503, db_down),
            ("ошибка внутри модели", 500, internal),
            ("by-weekday page=99", 404, await client.get("/predictions/by-weekday", params={"weekday": "monday", "page": 99})),
            ("GET /queue-status", 200, await client.get("/queue-status")),
        ]

        failed = 0
        print(f"{'№':>2} | {'Запрос':<38} | Ожид. | Получ. | Результат")
        for i, (name, expected, response) in enumerate(cases, 1):
            ok = response.status_code == expected
            failed += not ok
            print(f"{i:>2} | {name:<38} | {expected:>5} | {response.status_code:>6} | {'OK' if ok else 'FAIL'}")

        print()
        print("POST /predict ->", created.json())
        print("  headers: X-Request-ID =", created.headers["x-request-id"], "| X-Process-Time =", created.headers["x-process-time"])
        print("Перегрузка очереди (300 одновременных /predict-queued):",
              {code: flood_codes.count(code) for code in sorted(set(flood_codes))})
        print("429 ->", next(r for r in flood if r.status_code == 429).json(),
              "| Retry-After:", next(r for r in flood if r.status_code == 429).headers.get("retry-after"))
        print("504 ->", timeout_single.json())
        print("500 ->", internal.json())
        print("503 (LLM) ->", llm_down.json())
        print("409 ->", (await client.get(f"/predictions/{batch_id}")).json())
        print("by-weekday meta ->", (await client.get("/predictions/by-weekday", params={"weekday": "monday", "page": 1, "per_page": 5})).json()["meta"])
        print("LLM answer ->", llm_ok.json()["answer"][:200])

        # Сравнение одиночного и dynamic batching инференса под нагрузкой
        n = 64
        for path in ("/predict", "/predict-queued"):
            started = time.perf_counter()
            responses = await asyncio.gather(*[client.post(path, json=farm(i)) for i in range(n)])
            elapsed = time.perf_counter() - started
            assert all(r.status_code == 200 for r in responses)
            print(f"{n} одновременных {path:<16}: {elapsed * 1000:7.1f} мс, throughput {n / elapsed:6.1f} req/s")

        print()
        print("X-Process-Time:")
        for name, response in [
            ("/health", await client.get("/health")),
            ("/predict", await client.post("/predict", json=VALID)),
            ("/predict-with-llm", with_llm),
        ]:
            print(f"  {name:<18} {response.headers['x-process-time']} с")

    assert failed == 0, f"{failed} тест(ов) не прошли"


if __name__ == "__main__":
    asyncio.run(main())
