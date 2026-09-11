"""Интерфейс адаптера модели.

Адаптер отвечает на три вопроса и больше ни на что: доступен ли он, уместен ли на этих
данных и как собрать estimator. Он не знает про фолды, метрики и leaderboard — это делает
его заменяемым и не даёт логике сравнения расползтись по моделям.

Два правила, нарушение которых делает сравнение нечестным или ломает машину:

* **адаптер никогда не исчезает молча.** Недоступный показывается со статусом и текстом,
  что именно поставить; неуместный — с причиной. Пустое место в leaderboard заставляет
  думать, что модель проиграла, хотя её просто не запускали;
* **число потоков приходит из бюджета.** `n_jobs=-1` внутри адаптера при параллельном
  запуске даёт «модели × все ядра» — ровно тот дефект, что был в AutoDataAnalysis (D-12).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from sklearn.base import BaseEstimator

from backend.models.context import DatasetContext, ResourceBudget
from backend.preprocessing.profiles import ProfileKey

Family = Literal["dummy", "linear", "tree_ensemble", "gbdt", "svm", "neighbors"]
Verdict = Literal["ok", "caution", "skip"]


@dataclass(frozen=True)
class Availability:
    available: bool
    #обязателен, когда адаптер недоступен: «модель не найдена» пользователю бесполезно
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Applicability:
    verdict: Verdict
    #обязательна для caution и skip: молчаливый пропуск неотличим от проигрыша
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostEstimate:
    """Грубая оценка порядка величины, а не прогноз.

    Нужна, чтобы предупредить «этот контендер займёт десятки минут» до запуска,
    а не после. Точность здесь не требуется и не обещается.
    """

    relative_units: float
    note: str = ""


@runtime_checkable
class ModelAdapter(Protocol):
    key: str
    label: str
    family: Family
    supported_tasks: frozenset[str]
    preprocessing_profile: ProfileKey
    supports_proba: bool
    is_baseline: bool

    def availability(self) -> Availability: ...

    def applicability(self, context: DatasetContext) -> Applicability: ...

    def default_params(self, context: DatasetContext) -> dict[str, Any]: ...

    def build(
        self, params: dict[str, Any], budget: ResourceBudget, seed: int
    ) -> BaseEstimator: ...

    def estimate_cost(self, context: DatasetContext) -> CostEstimate: ...


class BaseAdapter:
    """Общее поведение адаптеров: доступность через импорт и нейтральная применимость."""

    key: str = ""
    label: str = ""
    family: Family = "linear"
    supported_tasks: frozenset[str] = frozenset({"binary", "multiclass", "regression"})
    preprocessing_profile: ProfileKey = "scaled_onehot"
    supports_proba: bool = True
    is_baseline: bool = False
    #модуль, отсутствие которого делает адаптер недоступным; None — часть ядра
    requires_module: str | None = None
    install_hint: str = ""

    def availability(self) -> Availability:
        if self.requires_module is None:
            return Availability(available=True)

        import importlib.util

        if importlib.util.find_spec(self.requires_module) is not None:
            return Availability(available=True)

        return Availability(
            available=False,
            reason=(
                f"Пакет «{self.requires_module}» не установлен. {self.install_hint} "
                "Ядро ModelArena работает без него — это опциональный участник."
            ).strip(),
        )

    def applicability(self, context: DatasetContext) -> Applicability:  # noqa: ARG002
        return Applicability(verdict="ok")

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {}

    def estimate_cost(self, context: DatasetContext) -> CostEstimate:
        #единица — обучение линейной модели на этих данных; остальные адаптеры
        #переопределяют множитель
        return CostEstimate(relative_units=float(context.n_train_rows) / 1000.0)

    def build(
        self, params: dict[str, Any], budget: ResourceBudget, seed: int
    ) -> BaseEstimator:
        raise NotImplementedError(f"Адаптер {self.key} не умеет собирать estimator.")
