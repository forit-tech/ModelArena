"""Предсказания как первичный артефакт (D-6).

Из прогона сохраняется не «строчка leaderboard», а сами предсказания по каждой строке.
Всё остальное — метрики, head-to-head, разбор ошибок, порог — пересчитывается из них
без повторного обучения. Метрика, записанная без предсказаний, непроверяема: её нельзя
ни перепроверить другой формулой, ни сопоставить с чужой.

`__row_id__` здесь — позиция строки в **снимке датасета**, а не в отфильтрованной
таблице. Фолды строятся по строкам с известной целью, и если записать их позиции,
join двух контендеров совпадёт, а join с исходными данными молча съедет на число
выброшенных строк.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

ROW_ID = "__row_id__"
SPLIT_CV = "cv"
SPLIT_HOLDOUT = "holdout"
HOLDOUT_FOLD = -1
PROBA_PREFIX = "proba__"


class PredictionError(Exception):
    """Предсказания не прошли проверку. Несёт машинный код для карточки контендера."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FoldPredictions:
    row_ids: np.ndarray
    fold: int
    split: str
    y_true: np.ndarray
    y_pred: np.ndarray
    #в порядке canonical class_labels; None для регрессии и моделей без вероятностей
    proba: np.ndarray | None


def align_probabilities(
    proba: np.ndarray,
    model_classes: list[str],
    canonical_labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Переставить столбцы вероятностей в канонический порядок меток.

    Столбцы `predict_proba` идут в порядке `classes_` конкретной модели. Полагаться
    на то, что он совпадёт с `TaskSpec.class_labels`, нельзя: достаточно одному
    контендеру обучиться на фолде без редкого класса, и ранговые метрики у разных
    моделей начнут считаться по разным столбцам — расхождение при этом будет
    правдоподобным по величине и потому незаметным.

    Класс, отсутствовавший в обучающей части фолда, получает нулевой столбец: модель
    его не видела и приписать ему массу не может. Такие метки возвращаются вторым
    значением, чтобы результат мог об этом сказать, а не сделать вид, что всё полно.
    """
    if proba.ndim != 2:
        raise PredictionError(
            "invalid_predictions",
            f"Ожидалась матрица вероятностей, получено измерений: {proba.ndim}.",
        )

    if proba.shape[1] != len(model_classes):
        raise PredictionError(
            "invalid_predictions",
            f"Столбцов вероятностей {proba.shape[1]}, а классов у модели {len(model_classes)}.",
        )

    unknown = [label for label in model_classes if label not in set(canonical_labels)]

    if unknown:
        #модель вернула класс, которого нет в постановке задачи: молча выбросить столбец
        #значило бы получить вероятности, не сходящиеся к единице, без объяснения
        raise PredictionError(
            "invalid_predictions",
            f"Модель предсказывает классы вне постановки задачи: {sorted(unknown)}.",
        )

    position_by_label = {label: index for index, label in enumerate(model_classes)}
    aligned = np.zeros((proba.shape[0], len(canonical_labels)), dtype=np.float64)
    missing: list[str] = []

    for target_index, label in enumerate(canonical_labels):
        source_index = position_by_label.get(label)

        if source_index is None:
            missing.append(label)
            continue

        aligned[:, target_index] = proba[:, source_index]

    return aligned, missing


def validate_fold_predictions(
    predictions: FoldPredictions,
    *,
    expected_rows: int,
    task_type: str,
    class_labels: list[str],
) -> None:
    """Проверить форму и значения до записи артефакта.

    Ошибка формы, пойманная здесь, — это внятный отказ контендера. Она же, пропущенная
    дальше, превращается в метрику, посчитанную по съехавшему выравниванию: число будет
    правдоподобным и потому не вызовет вопросов.
    """
    if len(predictions.y_pred) != expected_rows:
        raise PredictionError(
            "invalid_predictions",
            f"Модель вернула {len(predictions.y_pred)} предсказаний на {expected_rows} строк.",
        )

    if len(predictions.row_ids) != expected_rows:
        raise PredictionError(
            "invalid_predictions",
            f"Идентификаторов строк {len(predictions.row_ids)} при {expected_rows} строках.",
        )

    if task_type == "regression":
        values = np.asarray(predictions.y_pred, dtype=np.float64)

        if not np.isfinite(values).all():
            count = int((~np.isfinite(values)).sum())
            raise PredictionError(
                "invalid_predictions",
                f"{count} предсказаний — NaN или бесконечность; метрика по ним была бы "
                "бессмысленной.",
            )

        return

    predicted = set(np.asarray(predictions.y_pred, dtype=object).tolist())
    outside = sorted(str(label) for label in predicted - set(class_labels))

    if outside:
        raise PredictionError(
            "invalid_predictions", f"Предсказаны метки вне постановки задачи: {outside}."
        )

    if predictions.proba is None:
        return

    proba = np.asarray(predictions.proba, dtype=np.float64)

    if proba.shape != (expected_rows, len(class_labels)):
        raise PredictionError(
            "invalid_predictions",
            f"Форма вероятностей {proba.shape}, ожидалась {(expected_rows, len(class_labels))}.",
        )

    if not np.isfinite(proba).all():
        raise PredictionError(
            "invalid_predictions", "Среди вероятностей есть NaN или бесконечность."
        )

    if proba.size and bool((proba < 0).any()):
        raise PredictionError(
            "invalid_predictions", "Среди вероятностей есть отрицательные значения."
        )


def to_frame(
    parts: list[FoldPredictions],
    *,
    task_type: str,
    class_labels: list[str],
) -> pl.DataFrame:
    """Собрать таблицу предсказаний в едином виде для всех контендеров."""
    if not parts:
        raise PredictionError("invalid_predictions", "Ни одного предсказания не получено.")

    row_ids = np.concatenate([part.row_ids for part in parts]).astype(np.int64)
    folds = np.concatenate(
        [np.full(len(part.row_ids), part.fold, dtype=np.int32) for part in parts]
    )
    splits: list[str] = []

    for part in parts:
        splits.extend([part.split] * len(part.row_ids))

    columns: dict[str, Any] = {ROW_ID: row_ids, "split": splits, "fold": folds}

    if task_type == "regression":
        columns["y_true"] = np.concatenate(
            [np.asarray(part.y_true, dtype=np.float64) for part in parts]
        )
        columns["y_pred"] = np.concatenate(
            [np.asarray(part.y_pred, dtype=np.float64) for part in parts]
        )
        return pl.DataFrame(columns)

    columns["y_true"] = [str(value) for part in parts for value in part.y_true]
    columns["y_pred"] = [str(value) for part in parts for value in part.y_pred]

    if all(part.proba is not None for part in parts):
        matrix = np.concatenate([np.asarray(part.proba, dtype=np.float64) for part in parts])

        for index, label in enumerate(class_labels):
            columns[f"{PROBA_PREFIX}{label}"] = matrix[:, index].astype(np.float32)

    return pl.DataFrame(columns)


@dataclass(frozen=True)
class CoverageReport:
    """Насколько out-of-fold покрытие соответствует протоколу."""

    covered_rows: int
    expected_rows: int
    duplicated_rows: int
    complete: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_rows": self.covered_rows,
            "expected_rows": self.expected_rows,
            "duplicated_rows": self.duplicated_rows,
            "complete": self.complete,
            "note": self.note,
        }


def verify_oof_coverage(
    frame: pl.DataFrame,
    *,
    expected_row_ids: np.ndarray,
    strategy: str,
) -> CoverageReport:
    """Проверить, что каждая строка получила ровно одно out-of-fold предсказание.

    Дубликат означает, что строка попала в проверочную часть дважды и её вклад в метрику
    удвоился. Пропуск означает, что метрика посчитана по подмножеству, о котором нигде
    не сказано.

    При скользящем окне полнота **не требуется и не является дефектом**: первый отрезок
    времени служит только обучением и проверочным никогда не бывает. Требовать здесь
    полного покрытия значило бы объявить корректный протокол сломанным.
    """
    observed = frame.filter(pl.col("split") == SPLIT_CV)[ROW_ID].to_numpy()
    unique, counts = np.unique(observed, return_counts=True)
    duplicated = int((counts > 1).sum())

    if duplicated:
        raise PredictionError(
            "invalid_predictions",
            f"{duplicated} строк получили больше одного out-of-fold предсказания: "
            "их вклад в метрику удвоился бы.",
        )

    expected = set(expected_row_ids.tolist())
    covered = set(unique.tolist())
    stray = covered - expected

    if stray:
        raise PredictionError(
            "invalid_predictions",
            f"{len(stray)} предсказаний относятся к строкам вне обучающего пула.",
        )

    complete = covered == expected

    if complete:
        note = "Каждая строка обучающего пула получила ровно одно out-of-fold предсказание."
    elif strategy == "forward_chaining":
        note = (
            f"Покрыто {len(covered)} строк из {len(expected)}: при скользящем окне самый "
            "ранний отрезок времени служит только обучением и проверочным не бывает."
        )
    else:
        raise PredictionError(
            "invalid_predictions",
            f"Out-of-fold предсказания получили {len(covered)} строк из {len(expected)}: "
            "метрика считалась бы по подмножеству без объяснения.",
        )

    return CoverageReport(
        covered_rows=len(covered),
        expected_rows=len(expected),
        duplicated_rows=duplicated,
        complete=complete,
        note=note,
    )
