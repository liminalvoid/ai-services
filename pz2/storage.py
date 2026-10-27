"""
Хранилище прогнозов: SQLite через асинхронный драйвер aiosqlite.

Все запросы параметризованные (?), пользовательские данные
не подставляются в SQL через f-строки.
"""

import logging
import math
from contextlib import asynccontextmanager

import aiosqlite

import config
from errors import ServiceUnavailableError


logger = logging.getLogger(__name__)

STORAGE_UNAVAILABLE = "Prediction storage is unavailable"

# Поля, которые возвращают endpoints чтения
SELECT_COLUMNS = """
    request_id,
    farm_id,
    risk_score,
    risk_level,
    recommendation,
    model_version,
    source,
    created_at
"""

INSERT_SQL = """
    INSERT INTO predictions (
        request_id,
        farm_id,
        risk_score,
        risk_level,
        recommendation,
        model_version,
        source
    )
    VALUES (?, ?, ?, ?, ?, ?, ?);
"""

# SQLite strftime('%w', ...): 0 = воскресенье, 1 = понедельник, ...
WEEKDAY_MAP = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6"
}


@asynccontextmanager
async def get_db():
    """
    Отдельное соединение на одну логическую операцию.
    Любая ошибка SQLite превращается в 503.
    """

    try:
        db = await aiosqlite.connect(
            str(config.DATABASE_PATH),
            timeout=config.DB_BUSY_TIMEOUT_MS / 1000
        )
    except (aiosqlite.Error, OSError) as error:
        logger.exception("Database connection error")
        raise ServiceUnavailableError(STORAGE_UNAVAILABLE) from error

    db.row_factory = aiosqlite.Row

    try:
        await db.execute(f"PRAGMA busy_timeout={config.DB_BUSY_TIMEOUT_MS};")
        yield db

    except aiosqlite.Error as error:
        logger.exception("Database error")
        raise ServiceUnavailableError(STORAGE_UNAVAILABLE) from error

    finally:
        await db.close()


async def init_db() -> None:
    """
    Создает таблицу predictions при старте приложения.
    WAL позволяет читателям и писателю не блокировать друг друга.
    """

    async with get_db() as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")

        # request_id не PRIMARY KEY: один batch-запрос
        # порождает несколько прогнозов с одним request_id
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_request_id "
            "ON predictions(request_id);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_risk_level "
            "ON predictions(risk_level);"
        )
        await db.commit()

    logger.info("SQLite initialized | path=%s", config.DATABASE_PATH)


def _row(result: dict, request_id: str, source: str) -> tuple:
    return (
        request_id,
        result["farm_id"],
        result["risk_score"],
        result["risk_level"],
        result["recommendation"],
        result["model_version"],
        source
    )


async def save_prediction(result: dict, source: str) -> None:
    """
    Сохраняет один прогноз. Возврат из функции означает,
    что COMMIT выполнен успешно.
    """

    async with get_db() as db:
        await db.execute(
            INSERT_SQL,
            _row(result, result["request_id"], source)
        )
        await db.commit()

    logger.info(
        "Prediction saved | source=%s | farm_id=%s",
        source,
        result["farm_id"]
    )


async def save_predictions_batch(request_id: str, items: list[dict]) -> None:
    """
    Сохраняет весь batch одной транзакцией:
    executemany() + один commit().
    """

    async with get_db() as db:
        await db.executemany(
            INSERT_SQL,
            [_row(item, request_id, "batch") for item in items]
        )
        await db.commit()

    logger.info("Batch saved | rows=%s", len(items))


async def list_predictions(
    limit: int,
    risk_level: str | None = None
) -> list[dict]:
    """
    Возвращает последние прогнозы (новые первыми),
    при необходимости фильтруя по уровню риска.
    """

    async with get_db() as db:
        if risk_level is None:
            cursor = await db.execute(
                f"SELECT {SELECT_COLUMNS} FROM predictions "
                "ORDER BY id DESC LIMIT ?;",
                (limit,)
            )
        else:
            cursor = await db.execute(
                f"SELECT {SELECT_COLUMNS} FROM predictions "
                "WHERE risk_level = ? ORDER BY id DESC LIMIT ?;",
                (risk_level, limit)
            )

        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


async def get_predictions_by_request_id(request_id: str) -> list[dict]:
    """
    Все прогнозы одного HTTP-запроса: для /predict — одна строка,
    для /batch-predict — несколько.
    """

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT {SELECT_COLUMNS} FROM predictions "
            "WHERE request_id = ? ORDER BY id;",
            (request_id,)
        )
        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


async def get_predictions_by_weekday_page(
    weekday: str,
    page: int,
    per_page: int
) -> tuple[list[dict], int, int]:
    """
    Offset-пагинация: страница N = LIMIT per_page OFFSET (N-1)*per_page.
    Общее число записей считается через COUNT(*).

    Возвращает (записи страницы, total_items, total_pages).
    """

    weekday_number = WEEKDAY_MAP[weekday]
    offset = (page - 1) * per_page

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT {SELECT_COLUMNS} FROM predictions "
            "WHERE strftime('%w', created_at) = ? "
            "ORDER BY id DESC LIMIT ? OFFSET ?;",
            (weekday_number, per_page, offset)
        )
        rows = await cursor.fetchall()

        cursor = await db.execute(
            "SELECT COUNT(*) FROM predictions "
            "WHERE strftime('%w', created_at) = ?;",
            (weekday_number,)
        )
        (total_items,) = await cursor.fetchone()

    total_pages = math.ceil(total_items / per_page)

    return [dict(row) for row in rows], total_items, total_pages


async def save_seed_predictions(items: list[dict]) -> None:
    """
    Сохраняет демонстрационные прогнозы с датой в прошлом
    (поле days_ago), чтобы проверить выборку по дням недели.
    """

    rows = [
        (*_row(item, item["request_id"], "seed"), f"-{item['days_ago']} days")
        for item in items
    ]

    async with get_db() as db:
        await db.executemany(
            """
            INSERT INTO predictions (
                request_id, farm_id, risk_score, risk_level,
                recommendation, model_version, source, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', ?));
            """,
            rows
        )
        await db.commit()

    logger.info("Demo predictions seeded | rows=%s", len(items))
