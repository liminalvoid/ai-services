"""
Логирование со сквозным request_id.

Middleware записывает request_id в ContextVar, а фильтр
добавляет его в каждую запись журнала, сделанную во время
обработки этого запроса, — без явной передачи в logger.
"""

import logging
from contextvars import ContextVar

# "-" — запись сделана вне HTTP-запроса (старт, batch worker)
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(request_id)s | %(message)s"
    )

    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIdFilter())
