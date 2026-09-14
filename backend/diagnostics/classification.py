"""Диагностика классификации: матрица ошибок, разбор по классам, калибровка.

Бинарная и многоклассовая задачи здесь **не притворяются одним случаем**. У бинарной есть
положительный класс, ложные срабатывания и пропуски; у многоклассовой их нет, зато есть
вопрос «какой класс с каким путается». Показывать одно вместо другого — значит рисовать
метрику, которой в задаче нет.

Метрика, для которой нет данных, не рисуется вовсе. Модель без вероятностей не получает
нулевую калибровку — она получает пометку «неприменимо».
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from backend.arena.predictions import PROBA_PREFIX

#сколько корзин уверенности показывать в разрезе калибровки
CONFIDENCE_BINS = 5


def classification_diagnostics(
    validation: pl.DataFrame,
    *,
    class_labels: list[str],
    positive_label: str | None,
) -> dict[str, Any]:
    """Разбор качества классификации по out-of-fold предсказаниям."""
    if validation.height == 0:
        return {
            "available": False,
            "reason": "Нет out-of-fold предсказаний: разбирать нечего.",
        }

    truth = [str(value) for value in validation["y_true"].to_list()]
    predicted = [str(value) for value in validation["y_pred"].to_list()]
    labels = list(class_labels) or sorted(set(truth) | set(predicted))
    binary = len(labels) == 2

    return {
        "available": True,
        "n_rows": validation.height,
        "labels": labels,
        "is_binary": binary,
        "positive_label": positive_label if binary else None,
        "confusion": _confusion(truth, predicted, labels),
        "per_class": _per_class(truth, predicted, labels),
        "confidence": _confidence(validation, labels, truth, predicted),
    }


def _confusion(truth: list[str], predicted: list[str], labels: list[str]) -> dict[str, Any]:
    """Матрица ошибок: строки — факт, столбцы — предсказание."""
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for actual, guess in zip(truth, predicted, strict=True):
        if actual in index and guess in index:
            matrix[index[actual], index[guess]] += 1

    #самые частые путаницы выносятся отдельно: на большой матрице глазами их не найти
    confusions = [
        {
            "actual": labels[row],
            "predicted": labels[column],
            "count": int(matrix[row, column]),
        }
        for row in range(len(labels))
        for column in range(len(labels))
        if row != column and matrix[row, column] > 0
    ]
    confusions.sort(key=lambda item: item["count"], reverse=True)

    return {
        "labels": labels,
        "rows": matrix.tolist(),
        "top_confusions": confusions[:10],
    }


def _per_class(truth: list[str], predicted: list[str], labels: list[str]) -> list[dict[str, Any]]:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )

    return [
        {
            "label": label,
            "precision": round(float(precision[position]), 6),
            "recall": round(float(recall[position]), 6),
            "f1": round(float(f1[position]), 6),
            "support": int(support[position]),
        }
        for position, label in enumerate(labels)
    ]


def _confidence(
    validation: pl.DataFrame,
    labels: list[str],
    truth: list[str],
    predicted: list[str],
) -> dict[str, Any]:
    """Насколько уверенность модели соответствует её правоте.

    Это не полноценная кривая калибровки, а разрез по корзинам уверенности: в каждой
    корзине видно, какую долю модель обещала и какую угадала. Расхождение означает,
    что вероятностям нельзя доверять как вероятностям — а именно на них опираются
    выбор порога и оценка риска.
    """
    columns = [f"{PROBA_PREFIX}{label}" for label in labels]

    if any(column not in validation.columns for column in columns):
        return {
            "available": False,
            #«неприменимо», а не нулевая калибровка: нечего измерять
            "reason": "Модель не выдаёт вероятностей, поэтому уверенность измерить нечем.",
        }

    matrix = validation.select(columns).to_numpy().astype(np.float64)

    if not matrix.size:
        return {"available": False, "reason": "Вероятностей нет."}

    confidence = matrix.max(axis=1)
    correct = np.array(
        [actual == guess for actual, guess in zip(truth, predicted, strict=True)], dtype=bool
    )
    edges = np.linspace(0.0, 1.0, CONFIDENCE_BINS + 1)
    bins: list[dict[str, Any]] = []

    for position in range(CONFIDENCE_BINS):
        low, high = edges[position], edges[position + 1]
        inside = (confidence >= low) & (
            confidence <= high if position == CONFIDENCE_BINS - 1 else confidence < high
        )
        count = int(inside.sum())

        if not count:
            continue

        bins.append(
            {
                "from": round(float(low), 4),
                "to": round(float(high), 4),
                "count": count,
                "mean_confidence": round(float(confidence[inside].mean()), 6),
                "accuracy": round(float(correct[inside].mean()), 6),
            }
        )

    overconfident = [
        item for item in bins if item["mean_confidence"] - item["accuracy"] > 0.1
    ]

    return {
        "available": True,
        "bins": bins,
        "note": (
            "В каждой корзине сравниваются средняя обещанная уверенность и доля верных "
            "ответов. Заметное превышение обещания над правотой означает, что вероятности "
            "завышены и порог по ним выбирать нельзя."
        ),
        "overconfident_bins": len(overconfident),
    }
