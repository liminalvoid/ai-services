"""
Очередь запросов и dynamic batching.

Клиент отправляет одиночный запрос, endpoint кладет его в
asyncio.Queue и ждет Future. Фоновый worker собирает batch
(до MAX_BATCH_SIZE объектов или MAX_BATCH_WAIT_MS),
вызывает модель один раз и раздает каждому Future его score.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import config
import model
from errors import QueueFullError
from schemas import FarmRequest


logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future


# Создается в start_worker(): очередь должна принадлежать
# event loop'у запущенного приложения
inference_queue: asyncio.Queue | None = None


def queue_size() -> int:
    return inference_queue.qsize() if inference_queue else 0


def enqueue(data: FarmRequest, request_id: str) -> asyncio.Future:
    """
    Кладет запрос в очередь. Если очередь заполнена — 429
    (backpressure): клиент должен повторить запрос позже.
    """

    future = asyncio.get_running_loop().create_future()

    try:
        inference_queue.put_nowait(InferenceTask(data, request_id, future))

    except asyncio.QueueFull as error:
        logger.warning(
            "Queue is full | queue_size=%s",
            inference_queue.qsize()
        )
        raise QueueFullError("Inference queue is full. Try again later.") from error

    logger.info("Request queued | queue_size=%s", inference_queue.qsize())

    return future


async def collect_batch() -> list[InferenceTask]:
    """
    Ждет первый запрос, затем добирает batch, пока не наберется
    MAX_BATCH_SIZE или не истечет MAX_BATCH_WAIT_MS.
    """

    batch = []

    while not batch:
        task = await inference_queue.get()
        if task.future.cancelled():
            # Клиент уже получил 504, считать не нужно
            inference_queue.task_done()
        else:
            batch.append(task)

    deadline = time.perf_counter() + config.MAX_BATCH_WAIT_MS / 1000

    while len(batch) < config.MAX_BATCH_SIZE:
        remaining = deadline - time.perf_counter()

        if remaining <= 0:
            break

        try:
            task = await asyncio.wait_for(inference_queue.get(), timeout=remaining)
        except TimeoutError:
            break

        if task.future.cancelled():
            inference_queue.task_done()
        else:
            batch.append(task)

    return batch


async def batch_worker() -> None:
    logger.info("Batch worker started")

    while True:
        batch = await collect_batch()

        logger.info(
            "Dynamic batch formed | size=%s | queue_size=%s | request_ids=%s",
            len(batch),
            inference_queue.qsize(),
            ",".join(task.request_id[:8] for task in batch)
        )

        try:
            scores = await asyncio.wait_for(
                model.model_predict_batch([task.data for task in batch]),
                timeout=config.INFERENCE_TIMEOUT_SECONDS
            )

            for task, score in zip(batch, scores):
                if not task.future.done():
                    task.future.set_result(score)

        except Exception as error:
            logger.exception("Batch inference failed")

            for task in batch:
                if not task.future.done():
                    task.future.set_exception(error)

        finally:
            # На каждый queue.get() — один task_done()
            for _ in batch:
                inference_queue.task_done()


def start_worker() -> asyncio.Task:
    global inference_queue
    inference_queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
    return asyncio.create_task(batch_worker())


async def stop_worker(worker: asyncio.Task) -> None:
    worker.cancel()

    try:
        await worker
    except asyncio.CancelledError:
        pass
