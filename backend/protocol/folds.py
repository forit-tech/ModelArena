"""Фактическое разбиение на фолды.

Нужно раньше обучения: без реальных границ проверки утечки вырождаются в гадание.
Дубликат внутри обучающей части безвреден, дубликат, пересёкший границу
train ↔ validation, — конкретный механизм завышения метрики (D-23). Отличить одно
от другого можно только по фактическим индексам.

Разбиение строится **один раз** и передаётся всем контендерам: одинаковые фолды —
это и есть техническая гарантия честности сравнения (D-7). `assignment_hash`
существует, чтобы это можно было проверить, а не пообещать.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from backend.core.errors import ValidationError
from backend.protocol.recommend import ProtocolProposal
from backend.tasks.spec import TaskSpec


@dataclass(frozen=True)
class FoldPlan:
    """Индексы строк по частям. Позиции — в снимке после удаления пустой цели."""

    holdout: np.ndarray
    train_pool: np.ndarray
    #обучающая часть хранится ЯВНО, а не выводится вычитанием: при скользящем окне
    #обучение — это только прошлое, а не «весь пул кроме проверочной части».
    #Вывод по исключению давал бы модели будущее и делал бы проверку бессмысленной
    folds: list[tuple[np.ndarray, np.ndarray]]
    assignment_hash: str
    strategy: str

    @property
    def n_splits(self) -> int:
        return len(self.folds)

    @property
    def validation_folds(self) -> list[np.ndarray]:
        return [validation for _, validation in self.folds]

    def training_for(self, fold: int) -> np.ndarray:
        return self.folds[fold][0]

    def to_dict(self) -> dict[str, Any]:
        #наружу уходит описание, а не индексы: они внутреннее дело backend
        return {
            "strategy": self.strategy,
            "n_splits": self.n_splits,
            "holdout_rows": len(self.holdout),
            "train_pool_rows": len(self.train_pool),
            "validation_rows": [len(fold) for fold in self.validation_folds],
            "assignment_hash": self.assignment_hash,
        }


def build_folds(frame: pl.DataFrame, spec: TaskSpec, protocol: ProtocolProposal) -> FoldPlan:
    """Построить фактическое разбиение по предложенному протоколу."""
    usable = frame.filter(pl.col(spec.target_column).is_not_null())

    if usable.height < protocol.n_splits + 1:
        raise ValidationError(
            f"Строк с известной целью ({usable.height}) меньше, чем нужно для "
            f"{protocol.n_splits} фолдов."
        )

    positions = np.arange(usable.height)
    target = usable[spec.target_column]
    groups = (
        usable[spec.group_column].cast(pl.String).fill_null("__null__").to_numpy()
        if spec.group_column
        else None
    )

    holdout, train_pool = _split_holdout(
        usable, spec=spec, protocol=protocol, positions=positions, target=target, groups=groups
    )
    folds = _split_folds(
        protocol, train_pool=train_pool, target=target, groups=groups, usable=usable, spec=spec
    )

    return FoldPlan(
        holdout=holdout,
        train_pool=train_pool,
        folds=folds,
        assignment_hash=_assignment_hash(holdout, folds),
        strategy=protocol.cv_splitter,
    )


def _split_holdout(
    usable: pl.DataFrame,
    *,
    spec: TaskSpec,
    protocol: ProtocolProposal,
    positions: np.ndarray,
    target: pl.Series,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split

    if protocol.holdout_strategy == "temporal" and spec.time_column:
        #в holdout уходит самый поздний хвост: обучать на будущем и проверять на прошлом нельзя
        order = _time_order(usable, spec.time_column)
        count = max(1, round(len(order) * protocol.holdout_size))
        return order[-count:], order[:-count]

    if protocol.holdout_strategy == "group" and groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=protocol.holdout_size, random_state=protocol.seed
        )
        train_index, test_index = next(splitter.split(positions, groups=groups))
        return positions[test_index], positions[train_index]

    if protocol.holdout_strategy == "stratified":
        labels = target.cast(pl.String).to_numpy()
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=protocol.holdout_size, random_state=protocol.seed
        )
        train_index, test_index = next(splitter.split(positions.reshape(-1, 1), labels))
        return positions[test_index], positions[train_index]

    train_index, test_index = train_test_split(
        positions, test_size=protocol.holdout_size, random_state=protocol.seed, shuffle=True
    )
    return test_index, train_index


def _split_folds(
    protocol: ProtocolProposal,
    *,
    train_pool: np.ndarray,
    target: pl.Series,
    groups: np.ndarray | None,
    usable: pl.DataFrame,
    spec: TaskSpec,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    pool = train_pool
    labels = target.cast(pl.String).to_numpy()[pool]
    pool_groups = groups[pool] if groups is not None else None
    features = pool.reshape(-1, 1)

    if protocol.cv_splitter == "forward_chaining" and spec.time_column:
        #скользящее окно: каждая следующая часть по времени проверяет модель, обученную на прошлом
        #порядок считается по ТОМУ ЖЕ кадру, что и индексы пула: фильтрация здесь дала бы
        #позиции в другой таблице, и фолды собрались бы из чужих строк
        order = _time_order(usable, spec.time_column)
        pool_set = set(pool.tolist())
        ordered_pool = np.array([index for index in order if index in pool_set], dtype=np.int64)
        chunks = np.array_split(ordered_pool, protocol.n_splits + 1)
        plan: list[tuple[np.ndarray, np.ndarray]] = []

        for position in range(1, len(chunks)):
            validation = chunks[position]

            if not len(validation):
                continue

            #обучение — только то, что было РАНЬШЕ проверочной части
            training = np.concatenate(chunks[:position]) if position else np.array([], dtype=np.int64)

            if len(training):
                plan.append((training, validation))

        return plan

    if protocol.cv_splitter == "stratified_group_k_fold" and pool_groups is not None:
        splitter = StratifiedGroupKFold(
            n_splits=protocol.n_splits, shuffle=protocol.shuffle, random_state=protocol.seed
        )
        return [
            (pool[train], pool[validation])
            for train, validation in splitter.split(features, labels, groups=pool_groups)
        ]

    if protocol.cv_splitter == "group_k_fold" and pool_groups is not None:
        splitter = GroupKFold(n_splits=protocol.n_splits)
        return [
            (pool[train], pool[validation])
            for train, validation in splitter.split(features, groups=pool_groups)
        ]

    if protocol.cv_splitter == "stratified_k_fold":
        splitter = StratifiedKFold(
            n_splits=protocol.n_splits, shuffle=protocol.shuffle, random_state=protocol.seed
        )
        return [(pool[train], pool[validation]) for train, validation in splitter.split(features, labels)]

    splitter = KFold(
        n_splits=protocol.n_splits, shuffle=protocol.shuffle, random_state=protocol.seed
    )
    return [(pool[train], pool[validation]) for train, validation in splitter.split(features)]


def _time_order(usable: pl.DataFrame, time_column: str) -> np.ndarray:
    """Порядок строк по времени. Нераспознанные даты идут ПЕРВЫМИ.

    numpy сортирует `NaT` в конец, то есть строка с неизвестным временем оказывалась бы
    самой свежей и уходила в holdout — модель проверялась бы на наблюдениях, о которых
    вообще неизвестно, когда они произошли. Безопасная трактовка обратная: неизвестное
    время — это прошлое, и такие строки допустимы только в обучении.
    """
    column = usable[time_column]

    if column.dtype == pl.String:
        column = column.str.to_datetime(strict=False)

    values = column.to_numpy()
    known = column.is_not_null().to_numpy()
    positions = np.arange(len(values))
    unknown_first = positions[~known]
    ordered_known = positions[known][np.argsort(values[known], kind="stable")]

    return np.concatenate([unknown_first, ordered_known]).astype(np.int64)


def _assignment_hash(
    holdout: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> str:
    """Отпечаток разбиения: одинаковые фолды у всех контендеров — гарантия честности (D-7).

    В хеш входит и **обучающая** часть, а не только проверочная. При скользящем окне
    обучение задаётся отдельно, и два плана с одинаковой валидацией, но разным прошлым —
    это разные протоколы. Хеш только по валидации назвал бы их одинаковыми.

    Каждый блок предваряется длиной: без этого разбиения [[1],[2,3]] и [[1,2],[3]]
    дали бы одинаковый поток байт — та же неоднозначность склейки, что в отпечатке датасета.
    """
    digest = hashlib.sha256()

    def absorb(label: bytes, indices: np.ndarray) -> None:
        ordered = np.sort(indices).astype("<i8")
        digest.update(label)
        digest.update(len(ordered).to_bytes(8, "little"))
        digest.update(ordered.tobytes())

    absorb(b"holdout", holdout)

    for position, (training, validation) in enumerate(folds):
        digest.update(position.to_bytes(4, "little"))
        absorb(b"train", training)
        absorb(b"validation", validation)

    return digest.hexdigest()
