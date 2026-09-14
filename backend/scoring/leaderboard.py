"""Таблица результатов и выбор чемпиона.

Таблица **пересчитывается из сохранённых предсказаний**, а не читается из карточки прогона.
Это и есть смысл predictions-first (D-6): цель сравнения можно поменять и получить другой
порядок, ничего не обучая заново. Порядок, зашитый в момент обучения, пришлось бы
переобучать при каждом вопросе «а если для нас важнее полнота».

Три правила, без которых таблица превращается в обычный argmax:

* **holdout в ранжировании не участвует.** Он считается и показывается, но выбор делается
  по кросс-валидации. Иначе holdout перестаёт быть независимой проверкой: модель, выбранная
  по нему, на нём же и подтверждается, и итоговое число завышено;
* **чемпион обязан превзойти точку отсчёта устойчиво.** Модель, чьё преимущество над
  постоянным ответом не держится на фолдах, чемпионом не объявляется вовсе. Лучшая строка
  таблицы при этом видна — но «первый в таблице» и «победитель» это разные утверждения;
* **неразличимые модели называются неразличимыми.** Если разница меньше собственного
  колебания по фолдам, порядок между ними определяется объявленным заранее правилом,
  и в объяснении сказано каким именно.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.arena.metrics import MetricSet, evaluate, label_of, read_predictions
from backend.arena.predictions import SPLIT_CV, SPLIT_HOLDOUT
from backend.arena.store import RunRecord, RunStore
from backend.scoring.compare import compare_per_fold
from backend.scoring.objective import TIE_BREAKER_LABELS, Objective

logger = logging.getLogger("modelarena.scoring")


@dataclass
class ConstraintStatus:
    description: str
    metric: str
    observed: float | None
    satisfied: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "metric": self.metric,
            "observed": self.observed,
            #None означает «проверить не удалось» — это не «не прошла»
            "satisfied": self.satisfied,
        }


@dataclass
class LeaderboardRow:
    contender_key: str
    label: str
    family: str
    is_baseline: bool
    rank: int | None
    score: float | None
    std: float | None
    per_fold: list[float | None]
    estimated_cost: float
    #все метрики кросс-валидации, а не только целевая: выбор по одной не должен скрывать,
    #что модель проигрывает по остальным
    cross_validated: list[dict[str, Any]]
    #holdout показывается рядом и помечен как не участвовавший в выборе
    holdout: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[ConstraintStatus] = field(default_factory=list)
    eligible: bool = True
    reason: str = ""
    versus_next: dict[str, Any] | None = None
    versus_baseline: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contender_key": self.contender_key,
            "label": self.label,
            "family": self.family,
            "is_baseline": self.is_baseline,
            "rank": self.rank,
            "score": self.score,
            "std": self.std,
            "per_fold": self.per_fold,
            "estimated_cost": self.estimated_cost,
            "cross_validated": self.cross_validated,
            "holdout": self.holdout,
            "constraints": [item.to_dict() for item in self.constraints],
            "eligible": self.eligible,
            "reason": self.reason,
            "versus_next": self.versus_next,
            "versus_baseline": self.versus_baseline,
        }


@dataclass
class Champion:
    contender_key: str | None
    label: str
    reason: str
    over_baseline: dict[str, Any] | None = None
    over_runner_up: dict[str, Any] | None = None
    decided_by_tie_breaker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contender_key": self.contender_key,
            "label": self.label,
            "reason": self.reason,
            "over_baseline": self.over_baseline,
            "over_runner_up": self.over_runner_up,
            "decided_by_tie_breaker": self.decided_by_tie_breaker,
        }


@dataclass
class Leaderboard:
    objective: dict[str, Any]
    rows: list[LeaderboardRow]
    champion: Champion
    baseline_key: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "rows": [row.to_dict() for row in self.rows],
            "champion": self.champion.to_dict(),
            "baseline_key": self.baseline_key,
            "notes": self.notes,
        }


def build_leaderboard(
    *, record: RunRecord, store: RunStore, objective: Objective
) -> Leaderboard:
    """Собрать таблицу и вынести вердикт о чемпионе."""
    task = record.spec.get("task") or {}
    contenders = {item.get("contender_key"): item for item in record.spec.get("contenders", [])}
    rows: list[LeaderboardRow] = []

    for item in record.contenders:
        cost = float(contenders.get(item.contender_key, {}).get("estimated_cost", 0.0))
        rows.append(
            _row_for(
                record=record,
                store=store,
                objective=objective,
                contender_key=item.contender_key,
                label=item.label,
                family=item.family,
                is_baseline=item.is_baseline,
                state=item.state,
                state_reason=item.error_message or item.selection_reason,
                task=task,
                cost=cost,
            )
        )

    baseline_key = next((row.contender_key for row in rows if row.is_baseline), None)
    baseline_row = next((row for row in rows if row.is_baseline), None)
    ranked = _rank(rows, objective)
    _attach_comparisons(ranked, baseline_row, objective)
    champion = _choose_champion(ranked, baseline_row, objective)

    return Leaderboard(
        objective=objective.to_dict(),
        rows=ranked + [row for row in rows if row not in ranked],
        champion=champion,
        baseline_key=baseline_key,
        notes=_notes(record, objective),
    )


def _row_for(
    *,
    record: RunRecord,
    store: RunStore,
    objective: Objective,
    contender_key: str,
    label: str,
    family: str,
    is_baseline: bool,
    state: str,
    state_reason: str,
    task: dict[str, Any],
    cost: float,
) -> LeaderboardRow:
    row = LeaderboardRow(
        contender_key=contender_key,
        label=label,
        family=family,
        is_baseline=is_baseline,
        rank=None,
        score=None,
        std=None,
        per_fold=[],
        estimated_cost=cost,
        cross_validated=[],
        eligible=False,
        reason=state_reason,
    )

    if state != "SUCCEEDED":
        #участник без результата не занимает последнее место с нулём: у него нет числа,
        #и место в таблице ему давать не за что
        row.reason = state_reason or f"Участник завершился со статусом {state}."
        return row

    try:
        frame = read_predictions(store.predictions_path(record.run_id, contender_key))
        sets = evaluate(
            frame,
            task_type=str(task.get("task_type", "")),
            class_labels=list(task.get("class_labels") or []),
            positive_label=task.get("positive_label"),
        )
    except (OSError, ValueError, KeyError) as error:
        #таблица строится из предсказаний; если их нет, строку нечем наполнить,
        #и притворяться, что участник просто плох, нельзя
        logger.warning("Предсказания %s не читаются: %s", contender_key, error)
        row.reason = (
            f"Предсказания не читаются ({type(error).__name__}), пересчитать метрики нечем."
        )
        return row

    cross = next((item for item in sets if item.split == SPLIT_CV), None)
    holdout = next((item for item in sets if item.split == SPLIT_HOLDOUT), None)
    row.cross_validated = cross.to_dict()["metrics"] if cross else []
    row.holdout = holdout.to_dict()["metrics"] if holdout else []

    if cross is None:
        row.reason = "Нет out-of-fold предсказаний: ранжировать не по чему."
        return row

    target = cross.by_key(objective.metric)

    if target is None or target.value is None:
        #отсутствующая метрика остаётся отсутствующей: ноль поставил бы участника
        #на последнее место так, будто он проиграл, а он просто не измерен
        row.reason = (
            f"Целевая метрика {label_of(objective.metric)} не посчитана"
            + (f": {target.note}" if target and target.note else ".")
        )
        return row

    row.score = target.value
    row.std = target.std
    row.per_fold = target.per_fold
    row.constraints = _check_constraints(cross, objective)
    failed = [item for item in row.constraints if item.satisfied is False]
    unknown = [item for item in row.constraints if item.satisfied is None]

    if failed:
        row.eligible = False
        row.reason = "Не выполнены условия: " + "; ".join(item.description for item in failed)
        return row

    if unknown:
        row.eligible = False
        row.reason = (
            "Условия проверить не удалось: "
            + "; ".join(item.description for item in unknown)
            + ". Это не отказ модели, но и допустить её к выбору нельзя."
        )
        return row

    row.eligible = True
    row.reason = ""
    return row


def _check_constraints(cross: MetricSet, objective: Objective) -> list[ConstraintStatus]:
    statuses: list[ConstraintStatus] = []

    for constraint in objective.constraints:
        metric = cross.by_key(constraint.metric)
        observed = metric.value if metric else None
        statuses.append(
            ConstraintStatus(
                description=constraint.describe(),
                metric=constraint.metric,
                observed=observed,
                satisfied=constraint.satisfied_by(observed),
            )
        )

    return statuses


def _rank(rows: list[LeaderboardRow], objective: Objective) -> list[LeaderboardRow]:
    """Упорядочить допущенные строки по цели, разрешая ничьи объявленными правилами."""
    eligible = [row for row in rows if row.eligible and row.score is not None]
    sign = -1.0 if objective.higher_is_better else 1.0

    def key(row: LeaderboardRow) -> tuple:
        parts: list[float | str] = [sign * float(row.score or 0.0)]

        for rule in objective.tie_breakers:
            if rule == "stability":
                #при равном среднем впереди тот, у кого результат ровнее по фолдам
                parts.append(row.std if row.std is not None else float("inf"))
            elif rule == "cost":
                parts.append(row.estimated_cost)
            else:
                parts.append(row.contender_key)

        return tuple(parts)

    ordered = sorted(eligible, key=key)

    for position, row in enumerate(ordered, start=1):
        row.rank = position

    return ordered


def _attach_comparisons(
    ranked: list[LeaderboardRow],
    baseline: LeaderboardRow | None,
    objective: Objective,
) -> None:
    for position, row in enumerate(ranked):
        if position + 1 < len(ranked):
            row.versus_next = _compare(row, ranked[position + 1], objective)

        if baseline is not None and baseline.contender_key != row.contender_key:
            row.versus_baseline = _compare(row, baseline, objective)


def _compare(
    leader: LeaderboardRow, trailing: LeaderboardRow, objective: Objective
) -> dict[str, Any] | None:
    comparison = compare_per_fold(
        leader_key=leader.contender_key,
        leader_folds=leader.per_fold,
        trailing_key=trailing.contender_key,
        trailing_folds=trailing.per_fold,
        metric=label_of(objective.metric),
        higher_is_better=objective.higher_is_better,
    )
    return comparison.to_dict() if comparison else None


def _choose_champion(
    ranked: list[LeaderboardRow],
    baseline: LeaderboardRow | None,
    objective: Objective,
) -> Champion:
    if not ranked:
        return Champion(
            contender_key=None,
            label="",
            reason=(
                "Ни один участник не допущен к выбору: причины перечислены в строках таблицы."
            ),
        )

    leader = ranked[0]

    if leader.is_baseline:
        return Champion(
            contender_key=None,
            label="",
            reason=(
                f"Лучший результат показала точка отсчёта «{leader.label}». Ни одна модель "
                f"не превзошла постоянного ответа, значит признаки в этой постановке "
                f"предсказательной силы не дали."
            ),
        )

    if baseline is None or baseline.score is None:
        return Champion(
            contender_key=None,
            label="",
            reason=(
                "Точка отсчёта не посчиталась, сравнивать не с чем. Превосходство модели "
                "нельзя подтвердить без неё."
            ),
        )

    against_baseline = compare_per_fold(
        leader_key=leader.contender_key,
        leader_folds=leader.per_fold,
        trailing_key=baseline.contender_key,
        trailing_folds=baseline.per_fold,
        metric=label_of(objective.metric),
        higher_is_better=objective.higher_is_better,
    )

    if against_baseline is None:
        return Champion(
            contender_key=None,
            label="",
            reason="Сравнить с точкой отсчёта не на чем: нет общих фолдов с обоими значениями.",
        )

    if not against_baseline.stable:
        return Champion(
            contender_key=None,
            label="",
            reason=(
                f"Чемпион не выбран. Лидер таблицы — «{leader.label}», но его превосходство "
                f"над точкой отсчёта не держится на разбиении: {against_baseline.explanation} "
                f"Объявить победителя значило бы выдать колебание за результат."
            ),
            over_baseline=against_baseline.to_dict(),
        )

    if against_baseline.mean_difference < objective.min_gain_over_baseline:
        return Champion(
            contender_key=None,
            label="",
            reason=(
                f"Чемпион не выбран. «{leader.label}» превосходит точку отсчёта на "
                f"{against_baseline.mean_difference:.4f}, а объявленный заранее минимальный "
                f"выигрыш — {objective.min_gain_over_baseline:g}."
            ),
            over_baseline=against_baseline.to_dict(),
        )

    runner_up = ranked[1] if len(ranked) > 1 else None
    against_runner_up = (
        compare_per_fold(
            leader_key=leader.contender_key,
            leader_folds=leader.per_fold,
            trailing_key=runner_up.contender_key,
            trailing_folds=runner_up.per_fold,
            metric=label_of(objective.metric),
            higher_is_better=objective.higher_is_better,
        )
        if runner_up is not None
        else None
    )
    decided_by = ""
    reason = (
        f"«{leader.label}» — чемпион по цели «{objective.describe()}» "
        f"{against_baseline.explanation}"
    )

    if against_runner_up is not None and not against_runner_up.stable:
        #разница со вторым местом в пределах колебания: порядок определён правилом,
        #и об этом надо сказать прямо, иначе первое место читается как превосходство
        decided_by = _which_tie_breaker(leader, runner_up, objective)
        reason += (
            f" От второго места «{runner_up.label}» он при этом неотличим: "
            f"{against_runner_up.explanation} Порядок между ними определён правилом — "
            f"{TIE_BREAKER_LABELS.get(decided_by, decided_by)}."
        )

    return Champion(
        contender_key=leader.contender_key,
        label=leader.label,
        reason=reason,
        over_baseline=against_baseline.to_dict(),
        over_runner_up=against_runner_up.to_dict() if against_runner_up else None,
        decided_by_tie_breaker=decided_by,
    )


def _which_tie_breaker(
    leader: LeaderboardRow, runner_up: LeaderboardRow | None, objective: Objective
) -> str:
    if runner_up is None:
        return ""

    if leader.score != runner_up.score:
        return "score"

    for rule in objective.tie_breakers:
        if rule == "stability" and leader.std != runner_up.std:
            return rule

        if rule == "cost" and leader.estimated_cost != runner_up.estimated_cost:
            return rule

        if rule == "key":
            return rule

    return ""


def _notes(record: RunRecord, objective: Objective) -> list[str]:
    notes = [
        "Ранжирование сделано по кросс-валидации. Holdout посчитан и показан рядом, "
        "но в выборе не участвовал: модель, выбранная по holdout, подтверждалась бы "
        "на тех же строках, и итоговое число оказалось бы завышенным.",
        "Сравнение моделей идёт по разнице на каждом фолде в отдельности. Это законно "
        "потому, что все участники обучались на одном разбиении: общая трудность фолда "
        "входит в оба значения и при вычитании уходит.",
        "Устойчивость — не статистический тест. Фолды одного датасета не являются "
        "независимыми наблюдениями, поэтому слой сообщает наблюдаемую картину, "
        "а не p-value.",
    ]

    if record.state != "SUCCEEDED":
        notes.append(
            f"Прогон завершился со статусом {record.state}: часть участников в таблице "
            f"отсутствует, и сравнение неполно."
        )

    if objective.constraints:
        notes.append(
            "Участники, не выполнившие условия, остаются в таблице, но к выбору "
            "первого места не допускаются."
        )

    return notes
