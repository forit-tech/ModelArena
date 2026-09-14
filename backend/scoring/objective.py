"""Что именно значит «лучше».

Ранжирование по «максимальной accuracy» — это не выбор модели, а выбор числа. Оно
не отвечает ни на один вопрос, который на самом деле задают: важнее ли пропустить
положительный случай или зря потревожить отрицательный; допустима ли модель, которая
в среднем лучше, но проваливается на одном фолде; стоит ли выигрыш в третьем знаке
того, что модель считается в сто раз дольше.

Поэтому цель объявляется **до** взгляда на таблицу и переносится в карточку прогона.
Цель, выбранная после того, как результаты увидены, — это подгонка под желаемого
победителя, и отличить её потом невозможно.

Ограничения здесь — не фильтр «покрасивее», а отсев по пригодности: модель, не дотянувшая
до требуемой полноты, не участвует в борьбе за первое место, но **остаётся в таблице**
с указанием, какое ограничение она не прошла.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from backend.arena.metrics import metric_keys, primary_metric
from backend.core.errors import ValidationError

Operator = Literal["gte", "lte"]

#правила разрешения ничьей объявляются заранее и применяются в этом порядке.
#Без них порядок двух неразличимых моделей определялся бы случайностью сортировки,
#а пользователь читал бы его как настоящее превосходство
DEFAULT_TIE_BREAKERS: tuple[str, ...] = ("stability", "cost", "key")

TIE_BREAKER_LABELS: dict[str, str] = {
    "stability": "меньше разброс между фолдами",
    "cost": "дешевле в обучении",
    "key": "алфавитный порядок ключа — последнее правило, нужное только для повторяемости",
}


@dataclass(frozen=True)
class Constraint:
    """Требование к модели, невыполнение которого снимает её с борьбы за первое место."""

    metric: str
    operator: Operator
    value: float
    #зачем это ограничение: без объяснения отсев выглядит произволом
    reason: str = ""

    def describe(self) -> str:
        sign = "не ниже" if self.operator == "gte" else "не выше"
        tail = f" ({self.reason})" if self.reason else ""
        return f"{self.metric} {sign} {self.value:g}{tail}"

    def satisfied_by(self, value: float | None) -> bool | None:
        """None означает «проверить не удалось» — это не то же самое, что «не прошла»."""
        if value is None:
            return None

        return value >= self.value if self.operator == "gte" else value <= self.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"description": self.describe()}


@dataclass(frozen=True)
class Objective:
    """Объявленная цель сравнения."""

    metric: str
    task_type: str
    constraints: list[Constraint] = field(default_factory=list)
    tie_breakers: tuple[str, ...] = DEFAULT_TIE_BREAKERS
    #на сколько результат должен превзойти точку отсчёта, чтобы считаться содержательным.
    #Ноль означает «любое превосходство годится», и это осознанный выбор, а не умолчание
    min_gain_over_baseline: float = 0.0

    def __post_init__(self) -> None:
        allowed = metric_keys(self.task_type)

        if self.metric not in allowed:
            raise ValidationError(
                f"Метрика «{self.metric}» не считается для задачи {self.task_type}. "
                f"Доступны: {', '.join(allowed)}."
            )

        for constraint in self.constraints:
            if constraint.metric not in allowed:
                raise ValidationError(
                    f"Ограничение ссылается на метрику «{constraint.metric}», "
                    f"которая не считается для задачи {self.task_type}."
                )

        unknown = [name for name in self.tie_breakers if name not in TIE_BREAKER_LABELS]

        if unknown:
            raise ValidationError(f"Неизвестные правила разрешения ничьей: {unknown}.")

    @property
    def higher_is_better(self) -> bool:
        from backend.arena.metrics import direction_of

        return direction_of(self.metric) == "higher"

    def describe(self) -> str:
        head = (
            f"Максимизируем {self.metric}"
            if self.higher_is_better
            else f"Минимизируем {self.metric}"
        )
        if self.constraints:
            head += " при условиях: " + "; ".join(
                item.describe() for item in self.constraints
            )

        parts = [head]

        if self.min_gain_over_baseline:
            parts.append(
                f"Превосходство над точкой отсчёта должно быть не меньше "
                f"{self.min_gain_over_baseline:g}"
            )

        return ". ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "task_type": self.task_type,
            "higher_is_better": self.higher_is_better,
            "constraints": [item.to_dict() for item in self.constraints],
            "tie_breakers": [
                {"key": name, "rule": TIE_BREAKER_LABELS[name]} for name in self.tie_breakers
            ],
            "min_gain_over_baseline": self.min_gain_over_baseline,
            "description": self.describe(),
        }


def default_objective(task_type: str) -> Objective:
    """Цель по умолчанию для типа задачи.

    Объявлена заранее и одна на весь прогон. Подбирать метрику после того, как таблица
    увидена, — значит выбирать не модель, а метрику, при которой нужная модель выигрывает.
    """
    return Objective(metric=primary_metric(task_type), task_type=task_type)


def build_objective(
    task_type: str,
    *,
    metric: str | None = None,
    constraints: list[dict[str, Any]] | None = None,
    min_gain_over_baseline: float = 0.0,
) -> Objective:
    parsed: list[Constraint] = []

    for item in constraints or []:
        operator = item.get("operator", "gte")

        if operator not in {"gte", "lte"}:
            raise ValidationError(f"Неизвестное условие ограничения: {operator}.")

        try:
            value = float(item["value"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError("У ограничения отсутствует числовой порог.") from error

        parsed.append(
            Constraint(
                metric=str(item.get("metric", "")),
                operator=operator,
                value=value,
                reason=str(item.get("reason", "")),
            )
        )

    return Objective(
        metric=metric or primary_metric(task_type),
        task_type=task_type,
        constraints=parsed,
        min_gain_over_baseline=max(0.0, float(min_gain_over_baseline)),
    )
