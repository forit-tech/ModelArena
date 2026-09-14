"""Разбор ошибок: от «модель ошибается» к конкретным строкам.

Диагностика в числах отвечает, **насколько** модель ошибается. Этот слой отвечает,
**на чём именно** — и без него пользователь не может проверить, ошибается ли модель
на чепухе или на важных для него случаях.

`__row_id__` используется **только как ссылка на строку** и никогда не попадает
в признаки: он уникален почти на каждой строке, поэтому любая модель, получившая его,
выучила бы ответ наизусть и показала бы прекрасную метрику, ничего не обобщив.
Инвариант держится в `TaskSpec`, который отказывается принимать его признаком; здесь же
он живёт по своему прямому назначению — как адрес наблюдения.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

from backend.arena.predictions import PROBA_PREFIX, ROW_ID, SPLIT_CV
from backend.core.errors import ValidationError
from backend.datasets.registry import DatasetRegistry
from backend.diagnostics.thresholds import ERROR_ROWS_DEFAULT, ERROR_ROWS_MAX

ErrorKind = Literal[
    "all_mistakes",
    "false_positive",
    "false_negative",
    "confident_mistakes",
    "largest_error",
]


def select_error_rows(
    predictions: pl.DataFrame,
    *,
    dataset_id: str,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
    feature_columns: list[str],
    kind: ErrorKind = "all_mistakes",
    actual: str | None = None,
    predicted: str | None = None,
    limit: int = ERROR_ROWS_DEFAULT,
) -> dict[str, Any]:
    """Выбрать ошибочные строки и показать их вместе с исходными значениями признаков."""
    limit = max(1, min(int(limit), ERROR_ROWS_MAX))
    validation = predictions.filter(pl.col("split") == SPLIT_CV)

    if validation.height == 0:
        return {
            "kind": kind,
            "rows": [],
            "total_matching": 0,
            "note": "Нет out-of-fold предсказаний: разбирать нечего.",
        }

    selected, note = (
        _regression_errors(validation, kind=kind, limit=limit)
        if task_type == "regression"
        else _classification_errors(
            validation,
            kind=kind,
            actual=actual,
            predicted=predicted,
            class_labels=class_labels,
            positive_label=positive_label,
            limit=limit,
        )
    )
    total = selected.height
    head = selected.head(limit)

    return {
        "kind": kind,
        "total_matching": total,
        "shown": head.height,
        "limit": limit,
        "note": note,
        "rows": _join_features(
            head,
            dataset_id=dataset_id,
            class_labels=class_labels,
            feature_columns=feature_columns,
        ),
    }


def _classification_errors(
    validation: pl.DataFrame,
    *,
    kind: ErrorKind,
    actual: str | None,
    predicted: str | None,
    class_labels: list[str],
    positive_label: str | None,
    limit: int,
) -> tuple[pl.DataFrame, str]:
    del limit
    mistakes = validation.filter(pl.col("y_true") != pl.col("y_pred"))

    if kind == "confident_mistakes":
        column = f"{PROBA_PREFIX}{positive_label}" if positive_label else None
        probability_columns = [
            f"{PROBA_PREFIX}{label}"
            for label in class_labels
            if f"{PROBA_PREFIX}{label}" in validation.columns
        ]

        if not probability_columns:
            return mistakes, (
                "Модель не выдаёт вероятностей, поэтому «уверенные ошибки» выделить нечем: "
                "показаны все ошибочные строки."
            )

        del column
        confidence = (
            mistakes.select(probability_columns).to_numpy().max(axis=1)
            if mistakes.height
            else np.array([])
        )
        mistakes = mistakes.with_columns(pl.Series("confidence", confidence))
        return mistakes.sort("confidence", descending=True), (
            "Ошибки, в которых модель была увереннее всего. Именно они опаснее прочих: "
            "их не отсеять порогом."
        )

    if kind == "false_positive" and positive_label:
        return validation.filter(
            (pl.col("y_pred") == positive_label) & (pl.col("y_true") != positive_label)
        ), f"Ложные срабатывания: предсказан «{positive_label}», а на деле другой класс."

    if kind == "false_negative" and positive_label:
        return validation.filter(
            (pl.col("y_true") == positive_label) & (pl.col("y_pred") != positive_label)
        ), f"Пропуски: на деле «{positive_label}», а модель этого не увидела."

    if actual is not None and predicted is not None:
        return validation.filter(
            (pl.col("y_true") == actual) & (pl.col("y_pred") == predicted)
        ), f"Клетка матрицы ошибок: факт «{actual}», предсказание «{predicted}»."

    return mistakes, "Все строки, на которых предсказание не совпало с фактом."


def _regression_errors(
    validation: pl.DataFrame, *, kind: ErrorKind, limit: int
) -> tuple[pl.DataFrame, str]:
    del kind, limit
    with_error = validation.with_columns(
        (pl.col("y_pred") - pl.col("y_true")).alias("residual"),
        (pl.col("y_pred") - pl.col("y_true")).abs().alias("absolute_error"),
    )
    return with_error.sort("absolute_error", descending=True), (
        "Строки с наибольшей абсолютной ошибкой."
    )


def _join_features(
    rows: pl.DataFrame,
    *,
    dataset_id: str,
    class_labels: list[str],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Подставить исходные значения признаков по идентификатору строки.

    Снимок читается через реестр — тот же неизменяемый источник, на котором шло обучение.
    Читать что-то помимо него нельзя: иначе показанные значения могли бы относиться
    к другой версии данных, чем предсказание рядом с ними.
    """
    if rows.height == 0:
        return []

    registry = DatasetRegistry()

    try:
        snapshot = registry.load_frame(dataset_id, with_row_id=True)
    except Exception as error:
        raise ValidationError(
            f"Снимок датасета {dataset_id} не читается: показать исходные значения строк нечем."
        ) from error

    identifiers = rows[ROW_ID].to_list()
    features = snapshot.filter(pl.col(ROW_ID).is_in(identifiers))
    by_id = {int(item[ROW_ID]): item for item in features.iter_rows(named=True)}
    probability_columns = [
        f"{PROBA_PREFIX}{label}"
        for label in class_labels
        if f"{PROBA_PREFIX}{label}" in rows.columns
    ]
    prepared: list[dict[str, Any]] = []

    for item in rows.iter_rows(named=True):
        row_id = int(item[ROW_ID])
        source = dict(by_id.get(row_id) or {})
        source.pop(ROW_ID, None)
        wanted = set(feature_columns)
        entry: dict[str, Any] = {
            "row_id": row_id,
            "fold": int(item["fold"]),
            "y_true": item["y_true"],
            "y_pred": item["y_pred"],
            #только те колонки, которые модель действительно видела. Остальные, включая
            #целевую, идут отдельно: подписать цель словом «признак» значило бы показать
            #на экране разбора ошибок ровно то, чего у модели на входе не было
            "features": {
                key: _plain(value) for key, value in source.items() if key in wanted
            },
            "context": {
                key: _plain(value) for key, value in source.items() if key not in wanted
            },
        }

        if "absolute_error" in item:
            entry["absolute_error"] = round(float(item["absolute_error"]), 6)
            entry["residual"] = round(float(item["residual"]), 6)

        if probability_columns:
            entry["probabilities"] = {
                column.removeprefix(PROBA_PREFIX): round(float(item[column]), 6)
                for column in probability_columns
            }

        prepared.append(entry)

    return prepared


def _plain(value: Any) -> Any:
    """Привести значение к виду, пригодному для JSON, ничего не выдумывая."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return str(value)
