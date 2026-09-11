"""Технический профиль снимка: то, что видно о данных без выбора цели.

Профиль нужен карточке датасета и служит входом для Dataset Readiness. Всё, что зависит
от target, живёт не здесь — граница между DataArena и ModelArena проходит ровно по этому
признаку, и внутри ModelArena она тоже полезна.

**Никакого Quality Score.** Сводить состояние данных к одному числу — домен DataArena
(D-10); здесь показываются проверяемые величины, каждая со своим смыслом. Одна цифра
скрывает, какой именно сигнал её испортил, и провоцирует «улучшать метрику» вместо данных.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import polars as pl

#колонка почти уникальна — скорее всего технический идентификатор, а не признак
PROBABLE_ID_UNIQUE_RATIO = 0.98
#на короткой таблице почти любая колонка уникальна, и детекция теряет смысл
MIN_ROWS_FOR_ID_DETECTION = 10
#доля мажорного значения, при которой колонка перестаёт различать объекты
NEAR_CONSTANT_SHARE = 0.99
#сколько частых значений показывать в карточке категориальной колонки
TOP_VALUES_LIMIT = 5
EXAMPLES_LIMIT = 5


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    logical_type: str
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    is_near_constant: bool
    is_probable_id: bool
    id_reason: str | None
    numeric_stats: dict[str, float] | None
    top_values: list[dict[str, Any]] | None
    examples: list[Any]


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_row_count: int
    missing_cell_count: int
    missing_cell_ratio: float
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def column(self, name: str) -> ColumnProfile:
        return next(profile for profile in self.columns if profile.name == name)


def build_profile(frame: pl.DataFrame) -> DatasetProfile:
    """Посчитать профиль по снимку."""
    columns = [_profile_column(frame, name) for name in frame.columns]
    missing_cells = sum(profile.missing_count for profile in columns)
    total_cells = frame.height * frame.width

    return DatasetProfile(
        row_count=frame.height,
        column_count=frame.width,
        #дубли считаются по всей строке: именно они, попав в обе части разбиения,
        #превращают измерение обобщения в измерение памяти
        duplicate_row_count=frame.height - frame.n_unique(),
        missing_cell_count=missing_cells,
        missing_cell_ratio=_ratio(missing_cells, total_cells),
        columns=columns,
    )


def _profile_column(frame: pl.DataFrame, name: str) -> ColumnProfile:
    from backend.datasets.package import logical_type

    series = frame[name]
    row_count = frame.height
    missing = int(series.null_count())
    unique = int(series.n_unique())
    non_null = series.drop_nulls()
    logical = logical_type(series.dtype)
    is_probable_id, id_reason = _detect_probable_id(series, unique, row_count, missing)

    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        logical_type=logical,
        missing_count=missing,
        missing_ratio=_ratio(missing, row_count),
        unique_count=unique,
        unique_ratio=_ratio(unique, row_count),
        is_constant=unique <= 1,
        is_near_constant=_is_near_constant(non_null),
        is_probable_id=is_probable_id,
        id_reason=id_reason,
        numeric_stats=_numeric_stats(non_null) if logical in {"integer", "float", "decimal"} else None,
        top_values=_top_values(non_null) if logical in {"string", "boolean"} else None,
        examples=[_json_safe(value) for value in non_null.head(EXAMPLES_LIMIT).to_list()],
    )


def _detect_probable_id(
    series: pl.Series,
    unique_count: int,
    row_count: int,
    missing: int,
) -> tuple[bool, str | None]:
    #детекция отключается на коротких таблицах: на десяти строках почти всё уникально,
    #и относительный порог начинает называть идентификатором обычный признак
    if row_count < MIN_ROWS_FOR_ID_DETECTION:
        return False, None

    if missing or unique_count / row_count < PROBABLE_ID_UNIQUE_RATIO:
        return False, None

    if series.dtype.is_float():
        #дробное значение, уникальное почти для каждой строки, — это измерение, а не ключ
        return False, None

    if series.dtype.is_integer() and _is_monotonic_counter(series):
        return True, (
            f"Целые значения без пропусков идут подряд без разрывов на {row_count} строк — "
            "это автоинкремент, а не измеримая величина."
        )

    return True, (
        f"{unique_count} различных значений на {row_count} строк без пропусков: "
        "колонка почти не повторяется, поэтому как признак она бесполезна — "
        "на новых данных таких значений не будет."
    )


def _is_monotonic_counter(series: pl.Series) -> bool:
    #отличает технический счётчик от настоящей числовой величины
    ordered = series.drop_nulls().sort()

    if ordered.len() < 2:
        return False

    differences = ordered.diff().drop_nulls()
    return bool((differences == 1).all())


def _is_near_constant(non_null: pl.Series) -> bool:
    if non_null.len() == 0:
        return False

    counts = non_null.value_counts(sort=True)
    return bool(counts["count"].to_list()[0] / non_null.len() >= NEAR_CONSTANT_SHARE)


def _numeric_stats(non_null: pl.Series) -> dict[str, float] | None:
    if non_null.len() == 0:
        return None

    values = non_null.cast(pl.Float64)

    return {
        "min": _round(values.min()),
        "max": _round(values.max()),
        "mean": _round(values.mean()),
        "median": _round(values.median()),
        "std": _round(values.std()),
        "q1": _round(values.quantile(0.25)),
        "q3": _round(values.quantile(0.75)),
    }


def _top_values(non_null: pl.Series) -> list[dict[str, Any]] | None:
    if non_null.len() == 0:
        return None

    counts = non_null.value_counts(sort=True).head(TOP_VALUES_LIMIT)
    name = counts.columns[0]

    return [
        {
            "value": _json_safe(row[name]),
            "count": int(row["count"]),
            "ratio": _ratio(int(row["count"]), non_null.len()),
        }
        for row in counts.iter_rows(named=True)
    ]


def _ratio(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _round(value: float | None) -> float:
    return round(float(value), 6) if value is not None else 0.0


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return str(value)
