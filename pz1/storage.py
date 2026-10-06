predictions = {}


def save_prediction(result: dict) -> None:
    predictions[result["request_id"]] = result


def get_prediction(request_id: str) -> dict | None:
    return predictions.get(request_id)


def list_predictions(
    limit: int,
    risk_level: str | None = None
) -> list[dict]:
    """
    Возвращает последние прогнозы (новые первыми),
    при необходимости фильтруя по уровню риска.
    """

    # dict сохраняет порядок вставки,
    # поэтому reversed дает самые свежие записи первыми
    values = list(reversed(predictions.values()))

    if risk_level is not None:
        values = [
            item
            for item in values
            if item["risk_level"] == risk_level
        ]

    return values[:limit]
