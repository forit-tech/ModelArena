"""Диагностика регрессии: остатки, крупнейшие ошибки, ошибка по диапазонам цели.

Матрица ошибок и полнота здесь не имеют смысла, и рисовать их нельзя. Вопросы у регрессии
другие: смещена ли ошибка в одну сторону, растёт ли она с величиной цели, и где именно
модель промахивается сильнее всего.

Относительная ошибка считается **только там, где она определена**. При цели, близкой
к нулю, деление на неё даёт числа в тысячи процентов, которые выглядят как катастрофа,
а означают лишь маленький знаменатель.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

#сколько диапазонов цели показывать в разрезе ошибки
TARGET_BUCKETS = 5
#цель, меньшая этой доли от размаха, считается «около нуля»: относительная ошибка
#по ней не определена содержательно
NEAR_ZERO_SHARE = 0.01


def regression_diagnostics(validation: pl.DataFrame) -> dict[str, Any]:
    if validation.height == 0:
        return {"available": False, "reason": "Нет out-of-fold предсказаний: разбирать нечего."}

    truth = validation["y_true"].to_numpy().astype(np.float64)
    predicted = validation["y_pred"].to_numpy().astype(np.float64)
    residuals = predicted - truth

    return {
        "available": True,
        "n_rows": validation.height,
        "residuals": _residual_summary(residuals, truth),
        "by_target_range": _by_target_range(truth, residuals),
        "largest_errors": _largest(validation, residuals, truth),
    }


def _residual_summary(residuals: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    mean = float(residuals.mean())
    spread = float(residuals.std(ddof=0))
    #систематическое смещение: модель в среднем завышает или занижает. Заметить его
    #по RMSE нельзя — там знак теряется при возведении в квадрат
    biased = abs(mean) > 0.1 * spread if spread else abs(mean) > 0

    return {
        "mean": round(mean, 6),
        "std": round(spread, 6),
        "median": round(float(np.median(residuals)), 6),
        "min": round(float(residuals.min()), 6),
        "max": round(float(residuals.max()), 6),
        "quantiles": {
            str(level): round(float(np.quantile(residuals, level)), 6)
            for level in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "systematic_bias": bool(biased),
        "bias_note": (
            f"Средний остаток {mean:+.4f} при разбросе {spread:.4f}: модель систематически "
            + ("завышает" if mean > 0 else "занижает")
            + " предсказание. В RMSE это не видно — знак теряется при возведении в квадрат."
            if biased
            else "Систематического смещения не видно: средний остаток мал по сравнению с разбросом."
        ),
        "target_range": {
            "min": round(float(truth.min()), 6),
            "max": round(float(truth.max()), 6),
        },
    }


def _by_target_range(truth: np.ndarray, residuals: np.ndarray) -> list[dict[str, Any]]:
    """Ошибка по диапазонам цели: одинаково ли модель работает на всём размахе."""
    if truth.size < TARGET_BUCKETS * 2:
        return []

    edges = np.quantile(truth, np.linspace(0.0, 1.0, TARGET_BUCKETS + 1))
    buckets: list[dict[str, Any]] = []

    for position in range(TARGET_BUCKETS):
        low, high = edges[position], edges[position + 1]
        inside = (truth >= low) & (
            truth <= high if position == TARGET_BUCKETS - 1 else truth < high
        )
        count = int(inside.sum())

        if not count:
            continue

        part = residuals[inside]
        buckets.append(
            {
                "from": round(float(low), 6),
                "to": round(float(high), 6),
                "count": count,
                "mae": round(float(np.abs(part).mean()), 6),
                "rmse": round(float(np.sqrt((part**2).mean())), 6),
                "mean_residual": round(float(part.mean()), 6),
            }
        )

    return buckets


def _largest(
    validation: pl.DataFrame, residuals: np.ndarray, truth: np.ndarray
) -> list[dict[str, Any]]:
    """Строки с наибольшей абсолютной ошибкой."""
    limit = min(20, validation.height)
    order = np.argsort(-np.abs(residuals))[:limit]
    spread = float(truth.max() - truth.min())
    threshold = NEAR_ZERO_SHARE * spread if spread else 0.0
    rows: list[dict[str, Any]] = []
    identifiers = validation["__row_id__"].to_list()
    folds = validation["fold"].to_list()

    for position in order:
        actual = float(truth[position])
        #относительная ошибка только там, где знаменатель содержателен
        relative = (
            round(float(abs(residuals[position]) / abs(actual)), 6)
            if abs(actual) > threshold and actual != 0
            else None
        )
        rows.append(
            {
                "row_id": int(identifiers[position]),
                "fold": int(folds[position]),
                "y_true": round(actual, 6),
                "y_pred": round(float(truth[position] + residuals[position]), 6),
                "residual": round(float(residuals[position]), 6),
                "absolute_error": round(float(abs(residuals[position])), 6),
                "relative_error": relative,
            }
        )

    return rows
