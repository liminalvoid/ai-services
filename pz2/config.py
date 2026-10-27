"""
Эксплуатационные параметры сервиса.

Модули читают значения как config.NAME в момент вызова,
поэтому тесты могут временно менять их
(например, INFERENCE_TIMEOUT_SECONDS = 0.001).
"""

from pathlib import Path


# --- Dynamic / explicit batching ---

# Максимальное число объектов в одном batch
MAX_BATCH_SIZE = 8

# Максимальное время накопления dynamic batch
MAX_BATCH_WAIT_MS = 50

# Максимальное количество запросов в очереди (backpressure)
QUEUE_MAX_SIZE = 100

# Максимальное время ожидания результата инференса
INFERENCE_TIMEOUT_SECONDS = 2.0


# --- Имитация времени работы модели ---

# 20 мс — фиксированный overhead одного вызова модели,
# 3 мс — вычисления на один объект
MODEL_FIXED_OVERHEAD_SECONDS = 0.020
MODEL_PER_ITEM_SECONDS = 0.003


# --- SQLite ---

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "agro_scoring.db"

# Сколько ждать, если БД занята другой транзакцией
DB_BUSY_TIMEOUT_MS = 5000


# --- Внешняя LLM (Vireonix, OpenAI-совместимый API без ключа) ---

LLM_BASE_URL = "https://vireonix.ai/v1"
LLM_MODEL = "auto"
LLM_TIMEOUT_SECONDS = 60.0
LLM_DEFAULT_MAX_TOKENS = 300
LLM_DEFAULT_TEMPERATURE = 0.3
