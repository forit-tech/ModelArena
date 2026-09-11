"""Реестр адаптеров и сборка списка контендеров.

Список контендеров — это **весь** набор адаптеров, подходящих задаче, с явным статусом
у каждого. Ни один не выпадает молча: недоступный показывается с текстом, что поставить,
неуместный — с причиной. Пустое место в leaderboard читается как проигрыш, а не как
«не запускали», и это молчаливая неправда.

Baseline включается всегда и отключить его нельзя (ТЗ §8). Без него метрика 0.88 ничего
не значит: на датасете с 88% одного класса это результат постоянного ответа.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from backend.core.errors import ValidationError
from backend.models.adapters import boosters, core
from backend.models.base import ModelAdapter
from backend.models.context import DatasetContext, ResourceBudget
from backend.preprocessing.profiles import ProfileKey

ContenderStatus = Literal["ready", "caution", "skipped", "unavailable"]

#порядок фиксирован: он определяет порядок в интерфейсе и в отчётах,
#и не должен зависеть от порядка импорта
_ADAPTERS: tuple[ModelAdapter, ...] = (
    core.DummyClassifierAdapter(),
    core.DummyRegressorAdapter(),
    core.LogisticRegressionAdapter(),
    core.RidgeAdapter(),
    core.RandomForestClassifierAdapter(),
    core.RandomForestRegressorAdapter(),
    core.ExtraTreesClassifierAdapter(),
    core.ExtraTreesRegressorAdapter(),
    core.HistGradientBoostingClassifierAdapter(),
    core.HistGradientBoostingRegressorAdapter(),
    boosters.XgboostClassifierAdapter(),
    boosters.XgboostRegressorAdapter(),
    boosters.LightgbmClassifierAdapter(),
    boosters.LightgbmRegressorAdapter(),
    boosters.CatboostClassifierAdapter(),
    boosters.CatboostRegressorAdapter(),
    core.SvmClassifierAdapter(),
    core.KnnClassifierAdapter(),
)


@dataclass(frozen=True)
class Contender:
    """Участник сравнения вместе с причиной своего статуса."""

    contender_key: str
    adapter_key: str
    label: str
    family: str
    preprocessing_profile: ProfileKey
    supports_proba: bool
    is_baseline: bool
    status: ContenderStatus
    #обязательна для всего, кроме ready: иначе пропуск неотличим от проигрыша
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    cost_note: str = ""

    @property
    def runnable(self) -> bool:
        return self.status in {"ready", "caution"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adapters_for(task_type: str) -> list[ModelAdapter]:
    return [adapter for adapter in _ADAPTERS if task_type in adapter.supported_tasks]


def adapter_by_key(task_type: str, adapter_key: str) -> ModelAdapter:
    for adapter in adapters_for(task_type):
        if adapter.key == adapter_key:
            return adapter

    raise ValidationError(f"Адаптер «{adapter_key}» не найден для задачи {task_type}.")


def build_contenders(
    context: DatasetContext,
    *,
    selected_keys: list[str] | None = None,
) -> list[Contender]:
    """Собрать список контендеров со статусами.

    `selected_keys` ограничивает набор по выбору пользователя, но **не может убрать
    baseline**: без точки отсчёта leaderboard не читается.
    """
    contenders: list[Contender] = []

    for adapter in adapters_for(context.task_type):
        if selected_keys is not None and not adapter.is_baseline and adapter.key not in selected_keys:
            continue

        contenders.append(_describe(adapter, context))

    if not any(contender.is_baseline for contender in contenders):
        raise ValidationError(
            f"Для задачи {context.task_type} не нашлось baseline — сравнивать будет не с чем."
        )

    if not any(contender.runnable and not contender.is_baseline for contender in contenders):
        raise ValidationError(
            "Ни один контендер кроме baseline не может быть запущен на этих данных. "
            "Причины перечислены в статусах."
        )

    return contenders


def _describe(adapter: ModelAdapter, context: DatasetContext) -> Contender:
    availability = adapter.availability()

    if not availability.available:
        return Contender(
            contender_key=adapter.key,
            adapter_key=adapter.key,
            label=adapter.label,
            family=adapter.family,
            preprocessing_profile=adapter.preprocessing_profile,
            supports_proba=adapter.supports_proba,
            is_baseline=adapter.is_baseline,
            status="unavailable",
            reason=availability.reason,
        )

    applicability = adapter.applicability(context)
    status: ContenderStatus = (
        "skipped" if applicability.verdict == "skip" else applicability.verdict  # type: ignore[assignment]
    )
    cost = adapter.estimate_cost(context)

    return Contender(
        contender_key=adapter.key,
        adapter_key=adapter.key,
        label=adapter.label,
        family=adapter.family,
        preprocessing_profile=adapter.preprocessing_profile,
        supports_proba=adapter.supports_proba,
        is_baseline=adapter.is_baseline,
        status="ready" if applicability.verdict == "ok" else status,
        reason=applicability.reason,
        params=adapter.default_params(context),
        estimated_cost=round(cost.relative_units, 3),
        cost_note=cost.note,
    )


def build_estimator(
    task_type: str,
    contender: Contender,
    budget: ResourceBudget,
    seed: int,
):
    """Собрать estimator контендера. Seed и потоки приходят снаружи, а не изнутри модели."""
    if not contender.runnable:
        raise ValidationError(
            f"Контендер «{contender.label}» имеет статус {contender.status} и не запускается: "
            f"{contender.reason}"
        )

    adapter = adapter_by_key(task_type, contender.adapter_key)
    return adapter.build(dict(contender.params), budget, seed)
