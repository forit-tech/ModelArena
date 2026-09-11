"""Выбор протокола оценки: как разбивать данные и почему именно так.

Система предлагает протокол и **объясняет решение числами**, а не выбирает молча.
Это принципиально: случайное разбиение на данных со временем или с повторяющимися
сущностями завышает метрики, и пользователь узнаёт об этом только в продакшне.

Схема по умолчанию — CV на train pool плюс нетронутый holdout (D-1):

    весь датасет
       ├── train pool ──► K-fold CV ──► ранжирование, tuning, выбор Champion
       └── holdout    ──► один замер каждым контендером в конце

Ранжирование идёт по CV, а holdout остаётся независимым подтверждением. Здесь
протокол только **проектируется**; фактическое разбиение на индексы — следующий этап.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import polars as pl

from backend.core.errors import ValidationError
from backend.tasks.inference import find_group_column_candidates, find_time_column_candidates
from backend.tasks.spec import TaskSpec

CvSplitter = Literal[
    "stratified_k_fold", "k_fold", "group_k_fold", "stratified_group_k_fold", "forward_chaining"
]
HoldoutStrategy = Literal["stratified", "random", "group", "temporal"]

DEFAULT_N_SPLITS = 5
MIN_N_SPLITS = 2
MAX_N_SPLITS = 10
DEFAULT_HOLDOUT_SIZE = 0.2
MIN_HOLDOUT_SIZE = 0.1
MAX_HOLDOUT_SIZE = 0.4
DEFAULT_SEED = 42
#меньше этого числа строк в holdout доверительный интервал шире самой метрики
MIN_ROWS_FOR_INTERVALS = 30
#на такой выборке один фолд — это десяток строк, и разница между моделями тонет в шуме
SMALL_DATASET_ROWS = 50
#forward chaining требует нескольких различимых периодов, иначе это обычный holdout
MIN_PERIODS_FOR_FORWARD_CHAINING = 3


@dataclass(frozen=True)
class ProtocolProposal:
    cv_splitter: CvSplitter
    n_splits: int
    holdout_strategy: HoldoutStrategy
    holdout_size: float
    shuffle: bool
    seed: int
    explanation: str
    warnings: list[str] = field(default_factory=list)
    feasibility_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_protocol(
    frame: pl.DataFrame,
    spec: TaskSpec,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    holdout_size: float = DEFAULT_HOLDOUT_SIZE,
    seed: int = DEFAULT_SEED,
) -> ProtocolProposal:
    """Предложить протокол оценки для поставленной задачи."""
    _validate_request(n_splits, holdout_size)

    usable = frame.filter(pl.col(spec.target_column).is_not_null())
    warnings: list[str] = []
    notes: list[str] = []

    if spec.group_column:
        proposal = _group_protocol(usable, spec, n_splits, notes)
    elif spec.time_column:
        proposal = _temporal_protocol(usable, spec, n_splits, notes)
    elif spec.task_type in {"binary", "multiclass"}:
        proposal = _classification_protocol(usable, spec, n_splits, notes)
    else:
        proposal = _regression_protocol(n_splits)

    splitter, resolved_splits, holdout_strategy, explanation = proposal
    warnings.extend(_structural_warnings(usable, spec))
    warnings.extend(_size_warnings(usable, resolved_splits, holdout_size))

    return ProtocolProposal(
        cv_splitter=splitter,
        n_splits=resolved_splits,
        holdout_strategy=holdout_strategy,
        holdout_size=holdout_size,
        #перемешивание бессмысленно и вредно там, где порядок несёт смысл
        shuffle=splitter != "forward_chaining",
        seed=seed,
        explanation=explanation,
        warnings=warnings,
        feasibility_notes=notes,
    )


def _validate_request(n_splits: int, holdout_size: float) -> None:
    if not MIN_N_SPLITS <= n_splits <= MAX_N_SPLITS:
        raise ValidationError(f"Число фолдов должно быть от {MIN_N_SPLITS} до {MAX_N_SPLITS}.")

    if not MIN_HOLDOUT_SIZE <= holdout_size <= MAX_HOLDOUT_SIZE:
        raise ValidationError(
            f"Доля holdout должна быть от {MIN_HOLDOUT_SIZE:.0%} до {MAX_HOLDOUT_SIZE:.0%}: "
            "слишком маленький holdout не позволяет оценить модель, слишком большой "
            "оставляет мало данных на обучение."
        )


def _minority_class_count(frame: pl.DataFrame, target_column: str) -> tuple[int, str]:
    counts = frame[target_column].value_counts(sort=True)
    frequencies = counts["count"].to_list()
    labels = [str(value) for value in counts[target_column].to_list()]
    smallest = int(min(frequencies))
    return smallest, labels[frequencies.index(smallest)]


def _group_protocol(
    frame: pl.DataFrame,
    spec: TaskSpec,
    n_splits: int,
    notes: list[str],
) -> tuple[CvSplitter, int, HoldoutStrategy, str]:
    group_count = frame[spec.group_column].n_unique()

    if group_count < MIN_N_SPLITS:
        raise ValidationError(
            f"В колонке «{spec.group_column}» всего {group_count} различных сущностей — "
            "разбить по ним нельзя. Выберите другую колонку группировки или откажитесь от неё."
        )

    resolved = _reduce_splits(
        n_splits,
        group_count,
        notes,
        f"сущностей в «{spec.group_column}» всего {group_count}",
    )
    rows_per_group = frame.height / group_count
    explanation = (
        f"Наблюдения сгруппированы по «{spec.group_column}»: {group_count} сущностей "
        f"на {frame.height} строк, в среднем {rows_per_group:.1f} строки на сущность. "
        "Все строки одной сущности целиком попадают либо в обучение, либо в проверку — "
        "иначе модель узнаёт сущность, а не закономерность, и метрика оказывается завышенной."
    )

    if spec.task_type == "regression":
        return "group_k_fold", resolved, "group", explanation

    minority, label = _minority_class_count(frame, spec.target_column)

    if minority < resolved:
        notes.append(
            f"Стратификация внутри групп невозможна: класс «{label}» встречается {minority} раз, "
            f"это меньше числа фолдов ({resolved}). Используется group-разбиение без стратификации."
        )
        return "group_k_fold", resolved, "group", explanation

    return (
        "stratified_group_k_fold",
        resolved,
        "group",
        explanation + f" Доли классов при этом сохраняются: самый редкий встречается {minority} раз.",
    )


def _temporal_protocol(
    frame: pl.DataFrame,
    spec: TaskSpec,
    n_splits: int,
    notes: list[str],
) -> tuple[CvSplitter, int, HoldoutStrategy, str]:
    distinct_periods = frame[spec.time_column].n_unique()

    if distinct_periods < MIN_PERIODS_FOR_FORWARD_CHAINING:
        notes.append(
            f"В «{spec.time_column}» всего {distinct_periods} различных момента — "
            "для скользящего окна этого мало, применено обычное разбиение по фолдам."
        )
        return (
            "k_fold",
            n_splits,
            "temporal",
            f"Колонка «{spec.time_column}» задаёт порядок, но различных моментов слишком мало "
            "для полноценной валидации по времени. В holdout всё равно уходит последний по времени "
            "хвост: проверять модель на прошлом, обучив её на будущем, нельзя.",
        )

    resolved = _reduce_splits(
        n_splits,
        distinct_periods,
        notes,
        f"различных моментов в «{spec.time_column}» всего {distinct_periods}",
    )
    explanation = (
        f"Данные упорядочены по «{spec.time_column}» ({distinct_periods} различных моментов). "
        "Проверка идёт скользящим окном: модель обучается на прошлом и проверяется на будущем, "
        f"а в holdout уходит самый поздний хвост. Случайное разбиение здесь дало бы утечку "
        "из будущего в прошлое — модель знала бы то, чего в момент предсказания не существует."
    )
    return "forward_chaining", resolved, "temporal", explanation


def _classification_protocol(
    frame: pl.DataFrame,
    spec: TaskSpec,
    n_splits: int,
    notes: list[str],
) -> tuple[CvSplitter, int, HoldoutStrategy, str]:
    minority, label = _minority_class_count(frame, spec.target_column)

    if minority < MIN_N_SPLITS:
        raise ValidationError(
            f"Класс «{label}» встречается {minority} раз. Разбить его хотя бы на две части "
            "невозможно, поэтому честно оценить качество по нему нельзя. Объедините классы, "
            "соберите больше данных или выберите другую цель."
        )

    resolved = _reduce_splits(
        n_splits, minority, notes, f"самый редкий класс «{label}» встречается {minority} раз"
    )
    per_fold = minority / resolved
    explanation = (
        f"Задача классификации, самый редкий класс «{label}» — {minority} объектов "
        f"из {frame.height}. Стратифицированное разбиение сохраняет доли классов в каждом фолде, "
        f"поэтому в проверочной части оказывается около {per_fold:.1f} объектов этого класса, "
        "а не случайное их число."
    )

    if per_fold < MIN_N_SPLITS:
        notes.append(
            f"На фолд приходится всего ~{per_fold:.1f} объектов класса «{label}»: "
            "метрики по нему будут прыгать между фолдами, и разницу между моделями "
            "по нему сравнивать нельзя."
        )

    return "stratified_k_fold", resolved, "stratified", explanation


def _regression_protocol(n_splits: int) -> tuple[CvSplitter, int, HoldoutStrategy, str]:
    return (
        "k_fold",
        n_splits,
        "random",
        "Задача регрессии, структурных ограничений не обнаружено: строки разбиваются "
        "случайно на фолды с фиксированным seed. Это корректно, когда наблюдения независимы.",
    )


def _reduce_splits(requested: int, available: int, notes: list[str], reason: str) -> int:
    #понижаем число фолдов молча только в цифрах — причина обязана попасть в карточку,
    #иначе пользователь увидит «5 фолдов» в настройках и другое число в результатах
    if available >= requested:
        return requested

    resolved = max(MIN_N_SPLITS, available)
    notes.append(f"Запрошено {requested} фолдов, применено {resolved}: {reason}.")
    return resolved


def _structural_warnings(frame: pl.DataFrame, spec: TaskSpec) -> list[str]:
    #самая дорогая ошибка — не выбранная структура там, где она есть. Молчать об этом нельзя
    warnings: list[str] = []
    reserved = {spec.target_column}

    if not spec.time_column:
        candidates = find_time_column_candidates(frame.select(spec.feature_columns), reserved)

        if candidates:
            warnings.append(
                f"Похожие на время колонки не выбраны: {', '.join(candidates[:3])}. "
                "Если данные упорядочены во времени, случайное разбиение допустит утечку "
                "из будущего в прошлое, и метрика окажется оптимистичной."
            )

    if not spec.group_column:
        candidates = find_group_column_candidates(frame.select(spec.feature_columns), reserved)

        if candidates:
            warnings.append(
                f"Похожие на идентификатор сущности колонки не выбраны: {', '.join(candidates[:3])}. "
                "Если одна сущность даёт несколько строк, случайное разбиение разнесёт их "
                "по обучению и проверке, и модель будет узнавать сущность, а не закономерность."
            )

    return warnings


def _size_warnings(frame: pl.DataFrame, n_splits: int, holdout_size: float) -> list[str]:
    warnings: list[str] = []
    holdout_rows = round(frame.height * holdout_size)
    train_pool = frame.height - holdout_rows
    validation_rows = round(train_pool / n_splits) if n_splits else 0

    if frame.height < SMALL_DATASET_ROWS:
        warnings.append(
            f"Датасет небольшой: {frame.height} строк. На проверочную часть фолда придётся "
            f"около {validation_rows} строк, и разница между моделями будет в пределах шума."
        )

    if holdout_rows < MIN_ROWS_FOR_INTERVALS:
        warnings.append(
            f"В holdout попадёт около {holdout_rows} строк — меньше {MIN_ROWS_FOR_INTERVALS}. "
            "Доверительные интервалы на такой выборке шире самой метрики, поэтому "
            "подтверждение по holdout будет очень приблизительным."
        )

    return warnings
