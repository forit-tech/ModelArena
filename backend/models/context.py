"""Контекст, из которого адаптеры принимают решения о применимости.

Адаптер не должен знать про фолды, метрики и leaderboard — только про то, что за данные
перед ним и какими ресурсами он располагает. Всё, что нужно для решения «эта модель здесь
уместна или нет», собрано здесь и посчитано один раз.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from backend.readiness.report import estimate_effective_dimension
from backend.tasks.spec import TaskSpec

#оставляем ядро пользователю: инструмент не должен делать машину неотзывчивой
RESERVED_CORES = 1
DEFAULT_MEMORY_BUDGET_MB = 4096


@dataclass(frozen=True)
class ResourceBudget:
    """Сколько ресурсов адаптеру разрешено занять.

    Число потоков задаётся здесь, а не моделью: `n_jobs=-1` внутри адаптера при
    параллельном запуске нескольких контендеров даёт «модели × все ядра» и делает
    машину неотзывчивой. Это ровно тот дефект, который был в AutoDataAnalysis (D-12).
    """

    max_parallel_fits: int
    threads_per_fit: int
    memory_budget_mb: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def detect(cls, memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB) -> ResourceBudget:
        cores = max(1, (os.cpu_count() or 2) - RESERVED_CORES)
        #две модели параллельно по половине ядер: суммарно не больше доступного
        parallel = max(1, min(2, cores))
        return cls(
            max_parallel_fits=parallel,
            threads_per_fit=max(1, cores // parallel),
            memory_budget_mb=memory_budget_mb,
        )


@dataclass(frozen=True)
class DatasetContext:
    """Свойства данных, определяющие применимость модели."""

    task_type: str
    n_rows: int
    n_train_rows: int
    n_features: int
    effective_dimension: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    max_cardinality: int
    missing_ratio: float
    has_missing: bool
    n_classes: int | None
    min_class_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def estimated_matrix_bytes(self) -> int:
        #восемь байт на значение: столько занимает плотная матрица float64,
        #которую получит модель после кодирования
        return self.n_train_rows * self.effective_dimension * 8


def build_context(
    frame: pl.DataFrame,
    spec: TaskSpec,
    *,
    n_train_rows: int,
) -> DatasetContext:
    usable = frame.filter(pl.col(spec.target_column).is_not_null())
    selected = [name for name in spec.feature_columns if name in usable.columns]
    features = usable.select(selected) if selected else usable.select([])

    numeric = sum(
        1 for name in selected if features[name].dtype.is_numeric() and features[name].dtype != pl.Boolean
    )
    datetime = sum(1 for name in selected if features[name].dtype.is_temporal())
    categorical = len(selected) - numeric - datetime
    cardinalities = [
        features[name].n_unique()
        for name in selected
        if not features[name].dtype.is_numeric() or features[name].dtype == pl.Boolean
    ]
    missing_cells = sum(features[name].null_count() for name in selected)
    total_cells = max(1, usable.height * max(1, len(selected)))

    class_counts = (
        usable[spec.target_column].drop_nulls().value_counts()["count"].to_list()
        if spec.task_type != "regression"
        else []
    )

    return DatasetContext(
        task_type=spec.task_type,
        n_rows=usable.height,
        n_train_rows=n_train_rows,
        n_features=len(selected),
        effective_dimension=estimate_effective_dimension(usable, selected),
        n_numeric=numeric,
        n_categorical=categorical,
        n_datetime=datetime,
        max_cardinality=max(cardinalities) if cardinalities else 0,
        missing_ratio=round(missing_cells / total_cells, 6),
        has_missing=missing_cells > 0,
        n_classes=len(class_counts) if class_counts else None,
        min_class_count=min(class_counts) if class_counts else None,
    )
