# Лабораторная работа №2. Развитие AI-сервиса: request_id, асинхронная обработка, LLM

## Запуск

```bash
# Создание и запуск виртуальной среды для Python
python3 -m venv .venv && source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск приложения (Swagger UI: http://127.0.0.1:8000/docs)
uvicorn main:app --reload


# Запуск тестов (отдельная БД test_agro_scoring.db, кейсы с LLM обращаются к Vireonix)
python3 test_api.py
```

## Модули

| Модуль        | Ответственность                                                        |
|---------------|------------------------------------------------------------------------|
| `main.py`     | lifespan, middleware (request_id, X-Process-Time), обработчики ошибок, endpoints |
| `config.py`   | параметры batching, очереди, таймаутов, SQLite и LLM                   |
| `schemas.py`  | Pydantic-схемы (контракт API)                                          |
| `model.py`    | модель `calculate_risk()`, одиночный и пакетный вызов                  |
| `services.py` | бизнес-валидация, постпроцессинг, таймаут инференса                    |
| `batching.py` | `asyncio.Queue`, dynamic batching worker, backpressure                 |
| `storage.py`  | SQLite через `aiosqlite`: запись, batch-запись, поиск, пагинация       |
| `llm.py`      | клиент Vireonix LLM на `httpx.AsyncClient`                             |
| `errors.py`   | доменные ошибки и их HTTP-коды                                         |
| `logs.py`     | логирование с request_id в каждой записи                               |
