"""Пересечение границ разбиения.

Здесь проверяются **фактические** границы, а не наличие признака в датасете (D-23).
Разница принципиальна: дубликаты внутри обучающей части безвредны, а те же дубликаты,
разошедшиеся между обучением и проверкой, дают модели ответ на вопрос, который она
уже видела. То же с сущностями: если протокол групп-зависимый и группы изолированы,
находки нет — структура учтена, и жаловаться не на что.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from backend.leakage.report import LeakageSignal
from backend.protocol.folds import FoldPlan
from backend.protocol.recommend import ProtocolProposal
from backend.tasks.spec import TaskSpec

#разделитель между значениями и заглушка для пропуска. Оба обязаны не встречаться
#в данных: без разделителя строки ("ab", "c") и ("a", "bc") склеятся в одно значение,
#и разные наблюдения будут сочтены дубликатами — тот же класс ошибки, что в отпечатке
VALUE_SEPARATOR = "\x1f"
NULL_PLACEHOLDER = "\x00<null>"
#доля пересекающих строк, выше которой измерение перестаёт что-либо значить
BLOCKING_CROSSING_SHARE = 0.05


def check(
    usable: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,  # noqa: ARG001
    folds: FoldPlan,
) -> list[LeakageSignal]:
    signals = _duplicate_crossing(usable, spec, folds)
    signals.extend(_group_crossing(usable, spec, folds))
    signals.extend(_time_order_violation(usable, spec, folds))
    return signals


def _row_keys(usable: pl.DataFrame, spec: TaskSpec) -> np.ndarray:
    #строка вместе с ответом: одинаковые признаки при разных ответах — это шум,
    #а не утечка, и путать их нельзя
    columns = [*spec.feature_columns, spec.target_column]
    present = [name for name in columns if name in usable.columns]
    parts = []

    for name in present:
        column = pl.col(name)

        if usable[name].dtype.is_float():
            #-0.0 и 0.0 — одно значение по IEEE-754, но в строку они печатаются по-разному
            #и одинаковые строки перестали бы считаться дубликатами
            column = pl.when(column == 0).then(pl.lit(0.0)).otherwise(column)

        parts.append(column.cast(pl.String).fill_null(NULL_PLACEHOLDER))

    joined = usable.select(pl.concat_str(parts, separator=VALUE_SEPARATOR))
    return joined.to_series().to_numpy()


def _duplicate_crossing(usable: pl.DataFrame, spec: TaskSpec, folds: FoldPlan) -> list[LeakageSignal]:
    keys = _row_keys(usable, spec)
    crossings = 0
    affected_folds: list[int] = []

    for index, validation in enumerate(folds.validation_folds):
        training = folds.training_for(index)
        seen = set(keys[training].tolist())
        crossing = int(sum(1 for key in keys[validation] if key in seen))

        if crossing:
            crossings += crossing
            affected_folds.append(index)

    #множество строится ОДИН раз: внутри генератора оно пересоздавалось на каждой строке
    #holdout, и проверка становилась квадратичной — 60 секунд на 50 000 строк
    pool_keys = set(keys[folds.train_pool].tolist())
    holdout_crossing = int(sum(1 for key in keys[folds.holdout] if key in pool_keys))

    if not crossings and not holdout_crossing:
        return []

    total_validation = sum(len(fold) for fold in folds.validation_folds)
    share = crossings / total_validation if total_validation else 0.0

    return [
        LeakageSignal(
            code="duplicate_rows_cross_folds",
            evidence_level="structural",
            scope="protocol",
            columns=[],
            observed_value={
                "crossing_validation_rows": crossings,
                "crossing_holdout_rows": holdout_crossing,
                "affected_folds": affected_folds,
            },
            threshold="ни одна проверочная строка не должна дословно повторять обучающую",
            explanation=(
                f"{crossings} проверочных строк дословно повторяют обучающие вместе с ответом "
                f"(и {holdout_crossing} в holdout). Затронуто фолдов: {len(affected_folds)}."
            ),
            mechanism=(
                "На таких строках измеряется память, а не обобщение. Завышение достаётся прежде "
                "всего моделям с большой ёмкостью — именно тем, которые запоминают охотнее, — "
                "поэтому leaderboard систематически смещается в их пользу."
            ),
            suggested_action=(
                "Удалите полные дубли в DataArena либо, если повторы осмысленны, выберите "
                "разбиение по сущности — тогда копии останутся по одну сторону границы."
            ),
            blocking=share >= BLOCKING_CROSSING_SHARE,
        )
    ]


def _group_crossing(usable: pl.DataFrame, spec: TaskSpec, folds: FoldPlan) -> list[LeakageSignal]:
    if not spec.group_column or spec.group_column not in usable.columns:
        return []

    groups = usable[spec.group_column].cast(pl.String).fill_null("__null__").to_numpy()
    crossing_groups: set[str] = set()

    for index, validation in enumerate(folds.validation_folds):
        training = folds.training_for(index)
        crossing_groups |= set(groups[validation].tolist()) & set(groups[training].tolist())

    if not crossing_groups:
        #протокол групп-зависимый и группы изолированы — находки нет
        return []

    return [
        LeakageSignal(
            code="group_crosses_fold_boundary",
            evidence_level="structural",
            scope="protocol",
            columns=[spec.group_column],
            observed_value=len(crossing_groups),
            threshold="сущность целиком лежит по одну сторону границы фолда",
            explanation=(
                f"{len(crossing_groups)} сущностей из «{spec.group_column}» присутствуют "
                "одновременно в обучающей и в проверочной частях."
            ),
            mechanism=(
                "Модель узнаёт сущность по её же другим строкам, а не выучивает закономерность. "
                "Метрика отражает способность запоминать сущности, и на новых клиентах "
                "она не воспроизведётся."
            ),
            suggested_action=(
                "Выбранная колонка группировки не изолирует сущности — проверьте, та ли это "
                "колонка, и что протокол действительно групп-зависимый."
            ),
            blocking=True,
        )
    ]


def _time_order_violation(
    usable: pl.DataFrame, spec: TaskSpec, folds: FoldPlan
) -> list[LeakageSignal]:
    if not spec.time_column or spec.time_column not in usable.columns:
        return []

    column = usable[spec.time_column]

    if column.dtype == pl.String:
        column = column.str.to_datetime(strict=False)

    values = column.to_numpy()
    violations = 0

    for index, validation in enumerate(folds.validation_folds):
        training = folds.training_for(index)

        if not len(training) or not len(validation):
            continue

        #обучающая часть не имеет права содержать наблюдения позже проверочной
        if np.nanmax(values[training]) > np.nanmin(values[validation]):
            violations += 1

    if not violations:
        return []

    return [
        LeakageSignal(
            code="training_after_validation_in_time",
            evidence_level="structural",
            scope="protocol",
            columns=[spec.time_column],
            observed_value=violations,
            threshold="обучающая часть целиком раньше проверочной",
            explanation=(
                f"В {violations} фолдах обучающая часть содержит наблюдения позже проверочных "
                f"по «{spec.time_column}»."
            ),
            mechanism=(
                "Модель обучается на будущем и проверяется на прошлом. Она получает информацию, "
                "которой в момент реального предсказания не существует, и метрика окажется "
                "оптимистичной — сильнее у моделей, способных уловить временной тренд."
            ),
            suggested_action="Выберите разбиение по времени вместо текущего.",
            blocking=True,
        )
    ]
