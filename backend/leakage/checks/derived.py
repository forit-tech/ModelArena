"""Признак, полученный из цели.

Механизм самый прямой: если колонка содержит ответ, модель его переписывает.
Метрика становится идеальной и бессмысленной, а сравнение моделей превращается
в сравнение того, насколько быстро каждая нашла готовый ответ.

Здесь только **структурные** проверки: совпадение значений и точное восстановление
цели линейной функцией одного признака. Ни одна из них не опирается на корреляцию
как таковую — сильная корреляция сама по себе утечкой не является.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from backend.leakage.report import EXACT_MATCH_TOLERANCE, LeakageSignal
from backend.tasks.spec import TaskSpec, label_strings


def check(usable: pl.DataFrame, spec: TaskSpec) -> list[LeakageSignal]:
    signals = _exact_copies(usable, spec)

    if spec.task_type == "regression":
        signals.extend(_affine_reconstruction(usable, spec))

    return signals


def _exact_copies(usable: pl.DataFrame, spec: TaskSpec) -> list[LeakageSignal]:
    target_text = label_strings(usable[spec.target_column])
    matched: list[str] = []

    for name in spec.feature_columns:
        if name not in usable.columns:
            continue

        #сравнение по строковому представлению ловит и переименованную копию,
        #и колонку другого типа с теми же значениями
        if label_strings(usable[name]).equals(target_text):
            matched.append(name)

    if not matched:
        return []

    return [
        LeakageSignal(
            code="target_copy_among_features",
            evidence_level="structural",
            scope="features",
            columns=matched,
            observed_value=1.0,
            threshold="полное посимвольное совпадение значений с целью",
            explanation=(
                f"Признаки {', '.join(matched)} содержат ровно те же значения, что и цель "
                f"«{spec.target_column}»."
            ),
            mechanism=(
                "Модель переписывает ответ вместо того, чтобы его предсказывать. Все контендеры "
                "получат почти идеальную метрику, различия между ними станут случайным шумом, "
                "и leaderboard будет ранжировать не качество, а мелкие особенности реализации."
            ),
            suggested_action="Уберите эти колонки из признаков.",
            blocking=True,
        )
    ]


def _affine_reconstruction(usable: pl.DataFrame, spec: TaskSpec) -> list[LeakageSignal]:
    """Признак, из которого цель восстанавливается точно линейной функцией.

    Это не «высокая корреляция»: проверяется, что остаток после подгонки прямой
    равен нулю с точностью float. Честный предиктор, даже очень сильный, такого
    не даёт — у него остаётся собственная дисперсия.
    """
    target = usable[spec.target_column].cast(pl.Float64).to_numpy()

    if len(target) < 3 or not np.isfinite(target).all() or float(np.std(target)) == 0:
        return []

    reconstructed: list[dict[str, object]] = []

    for name in spec.feature_columns:
        if name not in usable.columns or not usable[name].dtype.is_numeric():
            continue

        values = usable[name].cast(pl.Float64).to_numpy()

        if not np.isfinite(values).all() or float(np.std(values)) == 0:
            continue

        slope, intercept = np.polyfit(values, target, 1)
        residual = float(np.max(np.abs(target - (slope * values + intercept))))
        scale = float(np.max(np.abs(target))) or 1.0

        if residual / scale < EXACT_MATCH_TOLERANCE:
            reconstructed.append({"column": name, "slope": round(float(slope), 6)})

    if not reconstructed:
        return []

    columns = [str(item["column"]) for item in reconstructed]

    return [
        LeakageSignal(
            code="target_derived_feature",
            evidence_level="structural",
            scope="features",
            columns=columns,
            observed_value=reconstructed,
            threshold="остаток линейной подгонки меньше 1e-9 от масштаба цели",
            explanation=(
                f"Цель «{spec.target_column}» восстанавливается из {', '.join(columns)} точной "
                "линейной функцией: остаток равен нулю с точностью float."
            ),
            mechanism=(
                "Такая связь означает, что колонка пересчитана из цели, а не измерена независимо. "
                "Любая модель найдёт её мгновенно, R² будет около единицы у всех, и сравнивать "
                "окажется нечего. Честный признак, даже очень сильный, сохраняет собственную "
                "дисперсию и точного восстановления не даёт."
            ),
            suggested_action=(
                "Проверьте происхождение колонки. Если она вычислена из цели — исключите её."
            ),
            blocking=True,
        )
    ]
