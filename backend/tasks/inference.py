"""Определение типа ML-задачи и кандидатов в служебные колонки.

Система **предлагает**, а не решает молча: у каждого вывода есть объяснение с числами,
и пользователь может его переопределить. Это то же правило, по которому работает выбор
стратегии валидации.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from backend.core.errors import ValidationError

TaskType = Literal["binary", "multiclass", "regression"]

#числовая колонка с небольшим числом повторяющихся значений — это классы, а не величина
MAX_CLASSIFICATION_CLASSES = 50
CLASSIFICATION_UNIQUE_SHARE = 0.05
#нижняя граница обязательна: без неё доля 5% на 100 строках даёт порог в 5 классов,
#и задача распознавания десяти цифр по сотне примеров была бы объявлена регрессией
MIN_CLASSIFICATION_CLASSES = 20
#потолок, выше которого колонка перестаёт быть целью классификации: 200 000 различных
#значений — это идентификатор или свободный текст, а не классы. Без потолка список меток
#целиком уходил бы в JSON-ответ и в карточку эксперимента
MAX_ALLOWED_CLASSES = 1_000
#временное разбиение имеет смысл только когда различных моментов достаточно
MIN_DISTINCT_TIMESTAMPS = 10
#сущность должна повторяться, иначе группы бессмысленны, но и не быть одной на весь датасет
MIN_GROUP_UNIQUE_RATIO = 0.02
MAX_GROUP_UNIQUE_RATIO = 0.6
MIN_GROUP_COUNT = 5
MIN_ROWS_FOR_GROUP_DETECTION = 20


@dataclass(frozen=True)
class TaskTypeProposal:
    task_type: TaskType
    class_labels: list[str]
    positive_label: str | None
    explanation: str
    warnings: list[str]


def infer_task_type(series: pl.Series) -> TaskTypeProposal:
    """Определить тип задачи по целевой колонке."""
    non_null = series.drop_nulls()
    dropped = series.len() - non_null.len()
    warnings: list[str] = []

    if non_null.len() == 0:
        raise ValidationError(f"Целевая колонка «{series.name}» пуста: предсказывать нечего.")

    if dropped:
        warnings.append(
            f"У {dropped} строк целевая колонка пуста — они не участвуют ни в обучении, ни в оценке."
        )

    if non_null.dtype.is_temporal():
        raise ValidationError(
            f"Колонка «{series.name}» содержит даты. Прогнозирование временных рядов "
            "в текущей версии не поддерживается — выберите другую цель."
        )

    unique_count = non_null.n_unique()

    if unique_count < 2:
        raise ValidationError(
            f"В колонке «{series.name}» одно значение на весь датасет. "
            "Модель не может научиться различать то, что не различается."
        )

    if _looks_categorical(non_null, unique_count):
        if unique_count > MAX_ALLOWED_CLASSES:
            raise ValidationError(
                f"В колонке «{series.name}» {unique_count} различных значений на "
                f"{non_null.len()} строк. Это идентификатор или свободный текст, а не классы: "
                f"классификация более чем на {MAX_ALLOWED_CLASSES} классов не имеет смысла — "
                "на большинство пришлось бы по несколько объектов. Выберите другую цель."
            )

        return _classification_proposal(series.name, non_null, unique_count, warnings)

    return TaskTypeProposal(
        task_type="regression",
        class_labels=[],
        positive_label=None,
        explanation=(
            f"«{series.name}» — числовая колонка с {unique_count} различными значениями "
            f"на {non_null.len()} строк. Это непрерывная величина, а не набор классов, "
            "поэтому задача решается как регрессия."
        ),
        warnings=warnings,
    )


def _looks_categorical(non_null: pl.Series, unique_count: int) -> bool:
    if non_null.dtype == pl.Boolean:
        return True

    if not non_null.dtype.is_numeric():
        return True

    #порог контекстный, но с полом: доля от размера выборки сверху ограничена 50 классами,
    #снизу — 20, иначе на коротких датасетах классификация превращалась бы в регрессию
    #на ровном месте. Решение остаётся за пользователем: это предложение, а не приговор
    limit = max(
        MIN_CLASSIFICATION_CLASSES,
        min(MAX_CLASSIFICATION_CLASSES, round(non_null.len() * CLASSIFICATION_UNIQUE_SHARE)),
    )
    return unique_count <= limit


def _classification_proposal(
    target_name: str,
    non_null: pl.Series,
    unique_count: int,
    warnings: list[str],
) -> TaskTypeProposal:
    counts = non_null.value_counts(sort=True)
    labels = [str(value) for value in counts[non_null.name].to_list()]
    frequencies = counts["count"].to_list()
    minority_count = int(min(frequencies))
    minority_label = labels[frequencies.index(minority_count)]

    if unique_count == 2:
        #положительным классом по умолчанию считается редкий: обычно интерес представляет
        #именно он — отток, дефолт, брак. Пользователь может поменять
        explanation = (
            f"«{target_name}» принимает два значения, поэтому это бинарная классификация. "
            f"Положительным классом предложен «{minority_label}» — он реже "
            f"({minority_count} из {non_null.len()}), а интерес обычно представляет редкое событие."
        )
        return TaskTypeProposal(
            task_type="binary",
            class_labels=sorted(labels),
            positive_label=minority_label,
            explanation=explanation,
            warnings=warnings,
        )

    if unique_count > MAX_CLASSIFICATION_CLASSES:
        warnings.append(
            f"Классов очень много ({unique_count}). На часть из них придётся по несколько строк, "
            "и метрики по ним будут неустойчивыми."
        )

    return TaskTypeProposal(
        task_type="multiclass",
        class_labels=sorted(labels),
        positive_label=None,
        explanation=(
            f"«{target_name}» принимает {unique_count} различных значений на {non_null.len()} строк — "
            f"это многоклассовая классификация. Самый редкий класс «{minority_label}»: "
            f"{minority_count} объектов."
        ),
        warnings=warnings,
    )


def find_time_column_candidates(frame: pl.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Колонки, пригодные для разбиения по времени.

    Проверяется не имя, а фактическая пригодность значений: колонка `runtime` не должна
    вводить в заблуждение, а `created` в виде строки — должна распознаваться.
    """
    excluded = exclude or set()
    candidates: list[str] = []

    for name in frame.columns:
        if name in excluded:
            continue

        series = frame[name]

        if series.dtype.is_temporal():
            if series.n_unique() >= MIN_DISTINCT_TIMESTAMPS:
                candidates.append(name)
            continue

        if series.dtype != pl.String:
            continue

        sample = series.drop_nulls().head(200)

        if sample.len() < MIN_DISTINCT_TIMESTAMPS:
            continue

        try:
            parsed = sample.str.to_datetime(strict=False)
        except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
            #на колонке вроде plan со значениями basic/pro разбор дат не находит формата.
            #Это не ошибка данных, а ответ «не кандидат»; ловим именно эти два типа,
            #чтобы настоящий сбой не был проглочен вместе с ними
            continue

        if parsed.null_count() / sample.len() <= 0.1 and parsed.n_unique() >= MIN_DISTINCT_TIMESTAMPS:
            candidates.append(name)

    return candidates


def find_group_column_candidates(frame: pl.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Колонки, похожие на идентификатор повторяющейся сущности.

    Именно такие требуют group-разбиения: наблюдения одной сущности не должны оказаться
    одновременно в train и в validation, иначе модель узнаёт сущность, а не закономерность.
    """
    excluded = exclude or set()
    row_count = frame.height

    if row_count < MIN_ROWS_FOR_GROUP_DETECTION:
        return []

    candidates: list[str] = []

    for name in frame.columns:
        if name in excluded:
            continue

        series = frame[name]

        if series.dtype.is_float() or series.dtype.is_temporal():
            continue

        unique_count = series.n_unique()

        if unique_count < MIN_GROUP_COUNT or unique_count >= row_count:
            continue

        unique_ratio = unique_count / row_count

        if MIN_GROUP_UNIQUE_RATIO <= unique_ratio <= MAX_GROUP_UNIQUE_RATIO:
            candidates.append(name)

    return candidates
