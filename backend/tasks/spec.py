"""Постановка ML-задачи: что предсказываем, чем и что исключено.

`TaskSpec` — воспроизводимое описание одного решения пользователя. Оно отделено
от протокола валидации намеренно: цель и признаки живут дольше, чем стратегия разбиения,
и меняются по разным причинам.

Каждое исключение признака несёт **причину и источник**. Через месяц вопрос
«почему эта колонка не участвовала» возникает обязательно, и ответ должен лежать
в карточке, а не в памяти того, кто запускал.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import polars as pl

from backend.core.errors import ValidationError
from backend.datasets.registry import ROW_ID, DatasetSnapshot
from backend.tasks.inference import (
    TaskType,
    find_group_column_candidates,
    find_time_column_candidates,
    infer_task_type,
)

ExclusionSource = Literal["user", "package_hint", "leakage_guard", "readiness", "reserved"]


def label_strings(series: pl.Series) -> pl.Series:
    """Каноническое строковое представление меток класса.

    Единая точка намеренно: `polars` приводит булеву колонку к «true»/«false»,
    а `str()` в Python — к «True»/«False». Если разные модули выберут разные
    представления, положительный класс перестанет находиться, и метрики будут
    считаться по пустому множеству, ничего при этом не сломав явно.
    """
    if series.dtype == pl.Boolean:
        return series.map_elements(
            lambda value: None if value is None else str(value), return_dtype=pl.String
        )

    return series.cast(pl.String)
#меньше этого числа строк любая оценка становится шумом, о котором нельзя молчать
MIN_TRAINING_ROWS = 20


@dataclass(frozen=True)
class ExclusionRecord:
    column: str
    reason: str
    source: ExclusionSource


@dataclass(frozen=True)
class TaskSpec:
    target_column: str
    task_type: TaskType
    feature_columns: list[str]
    excluded_columns: list[ExclusionRecord] = field(default_factory=list)
    positive_label: str | None = None
    class_labels: list[str] = field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSetupProposal:
    """Предложение системы до того, как пользователь что-то выбрал."""

    target_column: str
    task_type: TaskType
    class_labels: list[str]
    positive_label: str | None
    feature_columns: list[str]
    suggested_exclusions: list[ExclusionRecord]
    time_column_candidates: list[str]
    group_column_candidates: list[str]
    row_count: int
    usable_row_count: int
    explanation: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def propose_task_setup(
    snapshot: DatasetSnapshot,
    frame: pl.DataFrame,
    target_column: str,
) -> TaskSetupProposal:
    """Разобрать постановку задачи по выбранной цели, ничего не обучая."""
    _ensure_column_exists(frame, target_column)

    proposal = infer_task_type(frame[target_column])
    warnings = list(proposal.warnings)
    usable_rows = frame.height - frame[target_column].null_count()

    if usable_rows < MIN_TRAINING_ROWS:
        raise ValidationError(
            f"После удаления строк с пустой целью осталось {usable_rows} — "
            f"это меньше {MIN_TRAINING_ROWS}, оценка была бы бессмысленной."
        )

    exclusions = _suggest_exclusions(snapshot, frame, target_column)
    excluded_names = {record.column for record in exclusions}
    features = [
        name for name in frame.columns if name != target_column and name not in excluded_names
    ]

    if not features:
        raise ValidationError(
            "Не осталось ни одного признака: все колонки либо цель, либо предложены к исключению."
        )

    reserved = {target_column, *excluded_names}

    return TaskSetupProposal(
        target_column=target_column,
        task_type=proposal.task_type,
        class_labels=proposal.class_labels,
        positive_label=proposal.positive_label,
        feature_columns=features,
        suggested_exclusions=exclusions,
        time_column_candidates=find_time_column_candidates(frame, reserved),
        group_column_candidates=find_group_column_candidates(frame, reserved),
        row_count=frame.height,
        usable_row_count=usable_rows,
        explanation=proposal.explanation,
        warnings=warnings,
    )


def _suggest_exclusions(
    snapshot: DatasetSnapshot,
    frame: pl.DataFrame,
    target_column: str,
) -> list[ExclusionRecord]:
    #подсказки DataArena — advisory (D-9): они предлагаются к исключению, но проверяются
    #по фактическим данным, а не принимаются на веру
    records: list[ExclusionRecord] = []
    seen: set[str] = set()

    for column in snapshot.row_key:
        if column != target_column and column in frame.columns and column not in seen:
            seen.add(column)
            records.append(
                ExclusionRecord(
                    column=column,
                    reason=(
                        "Объявлен в пакете как row_key — устойчивая ссылка на строку. "
                        "Как признак он бесполезен: на новых данных таких значений не будет."
                    ),
                    source="package_hint",
                )
            )

    hints = (snapshot.package_ref or {}).get("hints") or {}

    for column in hints.get("identifier_columns") or []:
        if column == target_column or column not in frame.columns or column in seen:
            continue

        unique_ratio = frame[column].n_unique() / frame.height
        seen.add(column)
        records.append(
            ExclusionRecord(
                column=column,
                reason=(
                    f"DataArena отметила колонку как идентификатор; проверено по данным — "
                    f"{unique_ratio:.0%} значений уникальны."
                ),
                source="package_hint",
            )
        )

    if ROW_ID in frame.columns:
        records.append(
            ExclusionRecord(
                column=ROW_ID,
                reason="Служебный идентификатор строки ModelArena, признаком быть не может.",
                source="reserved",
            )
        )

    return records


def build_task_spec(
    frame: pl.DataFrame,
    *,
    target_column: str,
    task_type: TaskType,
    feature_columns: list[str],
    excluded_columns: list[ExclusionRecord] | None = None,
    positive_label: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
) -> TaskSpec:
    """Собрать проверенную постановку задачи.

    Проверки выполняются здесь целиком, до всякого обучения: пользователь получает
    понятную ошибку сразу, а не исключение sklearn через десять минут работы.
    """
    _ensure_column_exists(frame, target_column)

    if not feature_columns:
        raise ValidationError("Нужен хотя бы один признак помимо целевой колонки.")

    if target_column in feature_columns:
        raise ValidationError("Целевая колонка не может одновременно быть признаком.")

    if ROW_ID in feature_columns:
        raise ValidationError(f"{ROW_ID} — служебная колонка и признаком быть не может.")

    missing = [name for name in feature_columns if name not in frame.columns]

    if missing:
        raise ValidationError(f"В датасете нет признаков: {', '.join(missing)}.")

    duplicates = sorted({name for name in feature_columns if feature_columns.count(name) > 1})

    if duplicates:
        raise ValidationError(f"Признаки перечислены дважды: {', '.join(duplicates)}.")

    for role, column in (("группировки", group_column), ("времени", time_column)):
        if column is not None and column not in frame.columns:
            raise ValidationError(f"Колонка {role} «{column}» отсутствует в датасете.")

    class_labels = _resolve_class_labels(frame, target_column, task_type)
    positive = _resolve_positive_label(task_type, class_labels, positive_label)

    return TaskSpec(
        target_column=target_column,
        task_type=task_type,
        feature_columns=list(feature_columns),
        excluded_columns=list(excluded_columns or []),
        positive_label=positive,
        class_labels=class_labels,
        group_column=group_column,
        time_column=time_column,
    )


def _resolve_class_labels(frame: pl.DataFrame, target_column: str, task_type: TaskType) -> list[str]:
    #порядок меток фиксируется на весь прогон: по нему выравниваются столбцы вероятностей
    #всех контендеров, иначе ROC-AUC считался бы по разным колонкам у разных моделей
    if task_type == "regression":
        return []

    labels = sorted(set(label_strings(frame[target_column]).drop_nulls().to_list()))

    if task_type == "binary" and len(labels) != 2:
        raise ValidationError(
            f"Для бинарной классификации нужно ровно два класса, в «{target_column}» их {len(labels)}."
        )

    if task_type == "multiclass" and len(labels) < 2:
        raise ValidationError(f"В «{target_column}» меньше двух классов.")

    return labels


def _resolve_positive_label(
    task_type: TaskType,
    class_labels: list[str],
    positive_label: str | None,
) -> str | None:
    if task_type != "binary":
        return None

    if positive_label is None:
        raise ValidationError(
            "Для бинарной классификации нужно указать положительный класс: от него зависят "
            "precision, recall и PR-AUC."
        )

    if positive_label not in class_labels:
        raise ValidationError(
            f"Положительный класс «{positive_label}» отсутствует в целевой колонке. "
            f"Доступны: {', '.join(class_labels)}."
        )

    return positive_label


def _ensure_column_exists(frame: pl.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise ValidationError(f"Колонка «{column}» отсутствует в датасете.")
