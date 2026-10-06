from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

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

created = client.post("/predict", json=VALID)
request_id = created.json()["request_id"]

# Еще два прогноза разного уровня, чтобы проверить фильтрацию
client.post("/predict", json={**VALID, "farm_id": "FARM-002", "payment_delay_days": 0, "previous_defaults": 0, "debt": 0})
client.post("/predict", json={**VALID, "farm_id": "FARM-003", "previous_defaults": 0, "debt": 0})

cases = [
    ("GET /health", 200, client.get("/health")),
    ("GET /model-info", 200, client.get("/model-info")),
    ("корректный POST /predict", 201, created),
    ("area_ha = -100", 422, client.post("/predict", json={**VALID, "area_ha": -100})),
    ("неизвестный регион", 400, client.post("/predict", json={**VALID, "region": "Moscow"})),
    ("существующий request_id", 200, client.get(f"/predictions/{request_id}")),
    ("неизвестный request_id", 404, client.get("/predictions/abc-123")),
    ("limit = 2", 200, client.get("/predictions?limit=2")),
    ("risk_level = high", 200, client.get("/predictions?risk_level=high")),
    ("limit = -5", 422, client.get("/predictions?limit=-5")),
    # Дополнительные проверки
    ("limit = 101", 422, client.get("/predictions?limit=101")),
    ("risk_level = extreme", 422, client.get("/predictions?risk_level=extreme")),
    ("нет обязательного поля", 422, client.post("/predict", json={k: v for k, v in VALID.items() if k != "debt"})),
]

failed = 0
print(f"{'№':>2} | {'Запрос':<26} | Ожид. | Получ. | Результат")
for i, (name, expected, response) in enumerate(cases, 1):
    ok = response.status_code == expected
    failed += not ok
    print(f"{i:>2} | {name:<26} | {expected:>5} | {response.status_code:>6} | {'OK' if ok else 'FAIL'}")

print()
print("POST /predict ->", created.json())
print("limit=2 ->", len(client.get("/predictions?limit=2").json()), "записи")
print("risk_level=high ->", [p["farm_id"] for p in client.get("/predictions?risk_level=high").json()])
print("area_ha=-100 ->", client.post("/predict", json={**VALID, "area_ha": -100}).json())
print("X-Process-Time:", client.get("/health").headers.get("x-process-time"))

assert failed == 0, f"{failed} тест(ов) не прошли"
