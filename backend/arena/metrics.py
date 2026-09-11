"""Метрики считаются только из сохранённых предсказаний.

Ни один адаптер не считает «свою версию accuracy». Если каждая модель приносит
собственное число, сравнивать их нельзя: расхождение в усреднении, в положительном
классе или в обработке ничьей даёт разницу того же порядка, что и разница между
моделями, и leaderboard превращается в сравнение реализаций метрики.

Поэтому вход здесь — таблица предсказаний, и ничего кроме неё. Любое число на экране
можно пересчитать из артефакта заново, не запуская обучение (D-6).

Метрику, которую посчитать нельзя, слой возвращает как `value=None` с причиной.
Это отдельное состояние, а не ноль: ROC-AUC на фолде с одним классом не равен нулю,
он не определён, и подставленный ноль испортил бы среднее по фолдам.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from backend.arena.predictions import PROBA_PREFIX, ROW_ID, SPLIT_CV, SPLIT_HOLDOUT

#вероятности хранятся как float32, поэтому сумма по классам отличается от единицы
#на ошибку округления. Отклонение больше этого — уже не округление
PROBA_SUM_TOLERANCE = 1e-3

#метрика с этим направлением сортируется по убыванию, с "lower" — по возрастанию.
#Хранится рядом со значением, чтобы leaderboard не угадывал знак по имени метрики
Direction = str


@dataclass(frozen=True)
class MetricValue:
    key: str
    label: str
    direction: Direction
    #None означает «посчитать не удалось», а не ноль
    value: float | None
    std: float | None = None
    per_fold: list[float | None] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "direction": self.direction,
            "value": self.value,
            "std": self.std,
            "per_fold": self.per_fold,
            "note": self.note,
        }


@dataclass(frozen=True)
class MetricSet:
    split: str
    n_rows: int
    metrics: list[MetricValue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_rows": self.n_rows,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }

    def by_key(self, key: str) -> MetricValue | None:
        return next((metric for metric in self.metrics if metric.key == key), None)


_LABELS: dict[str, tuple[str, Direction]] = {
    "accuracy": ("Accuracy", "higher"),
    "balanced_accuracy": ("Balanced accuracy", "higher"),
    "f1": ("F1", "higher"),
    "precision": ("Precision", "higher"),
    "recall": ("Recall", "higher"),
    "roc_auc": ("ROC-AUC", "higher"),
    "pr_auc": ("PR-AUC", "higher"),
    "log_loss": ("Log loss", "lower"),
    "brier": ("Brier score", "lower"),
    "macro_f1": ("Macro F1", "higher"),
    "rmse": ("RMSE", "lower"),
    "mae": ("MAE", "lower"),
    "r2": ("R²", "higher"),
    "mape": ("MAPE", "lower"),
}


def metric_keys(task_type: str) -> list[str]:
    if task_type == "binary":
        return [
            "roc_auc",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "brier",
        ]

    if task_type == "multiclass":
        return ["macro_f1", "accuracy", "balanced_accuracy", "log_loss"]

    return ["rmse", "mae", "r2", "mape"]


def primary_metric(task_type: str) -> str:
    #ранжирование опирается на одну заранее объявленную метрику, а не на ту,
    #по которой победитель выглядит лучше
    if task_type == "binary":
        return "roc_auc"

    if task_type == "multiclass":
        return "macro_f1"

    return "rmse"


def evaluate(
    frame: pl.DataFrame,
    *,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
) -> list[MetricSet]:
    """Посчитать метрики по таблице предсказаний.

    Кросс-валидация усредняется **по фолдам**, а не по объединённым строкам (D-1):
    среднее по фолдам даёт разброс, по которому видно, устойчива разница между моделями
    или укладывается в шум. Склейка всех фолдов в одну кучу этот разброс уничтожает.
    """
    results: list[MetricSet] = []
    cv = frame.filter(pl.col("split") == SPLIT_CV)
    holdout = frame.filter(pl.col("split") == SPLIT_HOLDOUT)

    if cv.height:
        results.append(
            _cross_validated(
                cv, task_type=task_type, class_labels=class_labels, positive_label=positive_label
            )
        )

    if holdout.height:
        results.append(
            MetricSet(
                split=SPLIT_HOLDOUT,
                n_rows=holdout.height,
                metrics=_single(
                    holdout,
                    task_type=task_type,
                    class_labels=class_labels,
                    positive_label=positive_label,
                ),
            )
        )

    return results


def _cross_validated(
    cv: pl.DataFrame,
    *,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
) -> MetricSet:
    folds = sorted(cv["fold"].unique().to_list())
    per_fold: dict[str, list[float | None]] = {key: [] for key in metric_keys(task_type)}
    notes: dict[str, str] = {}

    for fold in folds:
        part = cv.filter(pl.col("fold") == fold)

        for metric in _single(
            part, task_type=task_type, class_labels=class_labels, positive_label=positive_label
        ):
            per_fold[metric.key].append(metric.value)

            if metric.note and metric.key not in notes:
                notes[metric.key] = metric.note

    metrics: list[MetricValue] = []

    for key in metric_keys(task_type):
        values = [value for value in per_fold[key] if value is not None]
        label, direction = _LABELS[key]
        skipped = len(per_fold[key]) - len(values)
        note = notes.get(key, "")

        if skipped and values:
            #среднее по неполному набору фолдов обязано об этом сказать: иначе оно
            #неотличимо от среднего по всем
            note = (
                f"Посчитано по {len(values)} фолдам из {len(per_fold[key])}. " + note
            ).strip()

        metrics.append(
            MetricValue(
                key=key,
                label=label,
                direction=direction,
                value=round(float(np.mean(values)), 6) if values else None,
                std=round(float(np.std(values, ddof=0)), 6) if len(values) > 1 else None,
                per_fold=per_fold[key],
                note=note or ("Не удалось посчитать ни на одном фолде." if not values else ""),
            )
        )

    return MetricSet(split=SPLIT_CV, n_rows=cv.height, metrics=metrics)


def _single(
    part: pl.DataFrame,
    *,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
) -> list[MetricValue]:
    if task_type == "regression":
        return _regression(part)

    return _classification(
        part, task_type=task_type, class_labels=class_labels, positive_label=positive_label
    )


def _regression(part: pl.DataFrame) -> list[MetricValue]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = part["y_true"].to_numpy().astype(np.float64)
    y_pred = part["y_pred"].to_numpy().astype(np.float64)
    metrics: list[MetricValue] = []

    metrics.append(_value("rmse", float(np.sqrt(mean_squared_error(y_true, y_pred)))))
    metrics.append(_value("mae", float(mean_absolute_error(y_true, y_pred))))

    if len(y_true) < 2 or float(np.var(y_true)) == 0.0:
        #R² сравнивает модель с предсказанием средним; при нулевой дисперсии сравнивать
        #не с чем, и формула делит на ноль
        metrics.append(
            _missing("r2", "Цель на этой части постоянна: R² не определён (деление на ноль).")
        )
    else:
        metrics.append(_value("r2", float(r2_score(y_true, y_pred))))

    zeros = int((y_true == 0).sum())

    if zeros:
        metrics.append(
            _missing(
                "mape",
                f"{zeros} строк с нулевой целью: относительная ошибка для них не определена.",
            )
        )
    else:
        metrics.append(_value("mape", float(np.mean(np.abs((y_true - y_pred) / y_true)))))

    return metrics


def _classification(
    part: pl.DataFrame,
    *,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
) -> list[MetricValue]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(part["y_true"].to_list(), dtype=object)
    y_pred = np.asarray(part["y_pred"].to_list(), dtype=object)
    proba, proba_note = _probabilities(part, class_labels)
    metrics: list[MetricValue] = []
    observed = set(y_true.tolist())

    metrics.append(_value("accuracy", float(accuracy_score(y_true, y_pred))))
    metrics.append(_value("balanced_accuracy", float(balanced_accuracy_score(y_true, y_pred))))

    if task_type == "multiclass":
        metrics.append(
            _value(
                "macro_f1",
                float(f1_score(y_true, y_pred, average="macro", labels=class_labels, zero_division=0)),
            )
        )
    else:
        positive = positive_label or (class_labels[-1] if class_labels else None)
        shared = {"labels": class_labels, "pos_label": positive, "zero_division": 0}
        metrics.append(_value("f1", float(f1_score(y_true, y_pred, **shared))))
        metrics.append(_value("precision", float(precision_score(y_true, y_pred, **shared))))
        metrics.append(_value("recall", float(recall_score(y_true, y_pred, **shared))))

        if proba is None:
            metrics.extend(_missing(key, proba_note) for key in ("roc_auc", "pr_auc", "brier"))
        elif len(observed) < 2:
            #на фолде с одним классом ранговая метрика не определена: ноль здесь был бы
            #не «плохим результатом», а вымышленным числом
            reason = f"На этой части присутствует один класс из {len(class_labels)}."
            metrics.extend(_missing(key, reason) for key in ("roc_auc", "pr_auc", "brier"))
        else:
            positive_index = class_labels.index(positive) if positive in class_labels else 1
            scores = proba[:, positive_index]
            binary_true = (y_true == positive).astype(int)
            metrics.append(_value("roc_auc", float(roc_auc_score(binary_true, scores))))
            metrics.append(_value("pr_auc", float(average_precision_score(binary_true, scores))))
            metrics.append(_value("brier", float(brier_score_loss(binary_true, scores))))

    if proba is None:
        metrics.append(_missing("log_loss", proba_note))
    elif len(observed) < len(class_labels):
        metrics.append(
            _missing(
                "log_loss",
                f"На этой части наблюдается {len(observed)} классов из {len(class_labels)}.",
            )
        )
    else:
        metrics.append(_value("log_loss", float(log_loss(y_true, proba, labels=class_labels))))

    order = metric_keys(task_type)
    return sorted(metrics, key=lambda metric: order.index(metric.key))


def _probabilities(
    part: pl.DataFrame, class_labels: list[str]
) -> tuple[np.ndarray | None, str]:
    """Матрица вероятностей в каноническом порядке меток либо None.

    Хранятся вероятности как float32 — вдвое компактнее при той же различимости.
    После чтения строки не суммируются к единице ровно, и `log_loss` на это ругается,
    а затем нормирует сам. Нормируем здесь и явно, но только в пределах точности float32:
    отклонение больше него — это не округление, а признак того, что столбцы выровнены
    неверно или часть массы потеряна, и такие вероятности возвращать нельзя.
    """
    columns = [f"{PROBA_PREFIX}{label}" for label in class_labels]

    if any(column not in part.columns for column in columns):
        return None, "Модель не выдаёт вероятностей."

    #порядок столбцов задаётся class_labels, а не порядком колонок в файле:
    #выравнивание вероятностей — единственное место, где ошибка даёт правдоподобное
    #и потому незаметное число
    matrix = part.select(columns).to_numpy().astype(np.float64)
    sums = matrix.sum(axis=1)

    if matrix.size and float(np.abs(sums - 1.0).max()) > PROBA_SUM_TOLERANCE:
        deviation = float(np.abs(sums - 1.0).max())
        return None, (
            f"Вероятности не суммируются к единице (отклонение до {deviation:.3g}): "
            "столбцы классов выровнены неверно или часть массы потеряна."
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(sums[:, None] > 0, matrix / sums[:, None], matrix)

    return normalized, ""


def _value(key: str, value: float) -> MetricValue:
    label, direction = _LABELS[key]
    return MetricValue(key=key, label=label, direction=direction, value=round(value, 6))


def _missing(key: str, note: str) -> MetricValue:
    label, direction = _LABELS[key]
    return MetricValue(key=key, label=label, direction=direction, value=None, note=note)


def read_predictions(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    required = {ROW_ID, "split", "fold", "y_true", "y_pred"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"В артефакте предсказаний нет колонок: {sorted(missing)}")

    return frame
