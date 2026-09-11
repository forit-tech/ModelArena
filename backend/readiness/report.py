"""Dataset Readiness: можно ли честно сравнивать модели на этих данных.

Вопрос, на который отвечает раздел, звучит не «насколько хороший датасет», а:

    можно ли честно сравнивать модели на этом датасете
    при **выбранной задаче** и **выбранном протоколе**?

Поэтому вход — тройка `DatasetSnapshot + TaskSpec + ProtocolProposal`, а не датасет
сам по себе (D-22). Проверка данных в отрыве от задачи — это data-quality dashboard,
и он остаётся в DataArena.

**Никакого балла от 0 до 100.** Одно число прячет причину, уравнивает несравнимое
и провоцирует улучшать метрику вместо данных. Вместо него — статус и список находок,
каждая из которых говорит, что измерено, с чем сравнивалось, что исказится и что делать.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import polars as pl

from backend.datasets.registry import DatasetSnapshot
from backend.protocol.recommend import ProtocolProposal
from backend.tasks.spec import TaskSpec

Severity = Literal["info", "caution", "high_risk"]
Status = Literal["READY", "CAUTION", "HIGH_RISK", "BLOCKED"]

#во сколько раз наблюдений должно быть больше, чем признаков после кодирования
COMFORTABLE_ROWS_PER_DIMENSION = 10
CRITICAL_ROWS_PER_DIMENSION = 2
#сколько объектов редкого класса на фолд ещё позволяет читать метрику
COMFORTABLE_MINORITY_PER_FOLD = 10
RISKY_MINORITY_PER_FOLD = 5
#доля мажорного класса, при которой accuracy перестаёт что-либо значить
IMBALANCE_CAUTION = 0.8
IMBALANCE_HIGH_RISK = 0.99
#строк в проверочной части фолда, ниже которых различия тонут в шуме
COMFORTABLE_VALIDATION_ROWS = 30
CRITICAL_VALIDATION_ROWS = 10
#доля пропусков в выбранном признаке
MISSING_CAUTION = 0.5
MISSING_HIGH_RISK = 0.9
#доля пропусков в цели
TARGET_MISSING_CAUTION = 0.2
TARGET_MISSING_HIGH_RISK = 0.5
#доля полных дублей строк
DUPLICATE_HIGH_RISK = 0.05
#предел one-hot из профиля препроцессинга: редкие уровни схлопываются
ONE_HOT_CAP = 50
#календарных признаков из одной даты
DATETIME_PARTS = 6


@dataclass(frozen=True)
class Finding:
    """Одна причина, по которой сравнение может оказаться нечестным.

    Ни одно поле не факультативно: находка без `consequence` не объясняет, чем она
    грозит, а без `suggested_action` учит игнорировать находки вообще.
    """

    code: str
    severity: Severity
    scope: str
    observed_value: Any
    threshold: str
    explanation: str
    consequence: str
    suggested_action: str
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessContext:
    """Числа, на которых строятся все выводы. Вынесены наружу, чтобы пороги
    можно было проверить, а не принимать на веру."""

    usable_rows: int
    total_rows: int
    feature_count: int
    effective_dimension: int
    rows_per_dimension: float
    n_splits: int
    holdout_rows: int
    validation_rows_per_fold: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessReport:
    status: Status
    summary: str
    context: ReadinessContext
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "context": self.context.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def by_code(self, code: str) -> Finding | None:
        return next((finding for finding in self.findings if finding.code == code), None)


def estimate_effective_dimension(frame: pl.DataFrame, feature_columns: list[str]) -> int:
    """Размерность **после кодирования**, а не число колонок.

    Считать признаки штуками бессмысленно: одна категориальная колонка на сорок уровней
    даёт сорок столбцов, а десять числовых — десять. Именно эта величина определяет,
    хватает ли наблюдений.
    """
    total = 0

    for name in feature_columns:
        if name not in frame.columns:
            continue

        dtype = frame[name].dtype

        if dtype.is_temporal():
            total += DATETIME_PARTS
        elif dtype.is_numeric() or dtype == pl.Boolean:
            total += 1
        else:
            #one-hot с ограничением: редкие уровни схлопываются в infrequent
            total += max(1, min(frame[name].n_unique(), ONE_HOT_CAP))

    return total


def build_context(
    frame: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
) -> ReadinessContext:
    usable = frame.height - frame[spec.target_column].null_count()
    holdout_rows = round(usable * protocol.holdout_size)
    train_pool = usable - holdout_rows
    effective = estimate_effective_dimension(frame, spec.feature_columns)

    return ReadinessContext(
        usable_rows=usable,
        total_rows=frame.height,
        feature_count=len(spec.feature_columns),
        effective_dimension=effective,
        rows_per_dimension=round(usable / effective, 2) if effective else 0.0,
        n_splits=protocol.n_splits,
        holdout_rows=holdout_rows,
        validation_rows_per_fold=round(train_pool / protocol.n_splits) if protocol.n_splits else 0,
    )


def resolve_status(findings: list[Finding]) -> Status:
    #BLOCKED — когда честное сравнение невозможно в принципе, а не когда результат
    #выглядит плохо. Это разные вещи, и путать их значит блокировать нормальную работу
    if any(finding.blocking for finding in findings):
        return "BLOCKED"

    if any(finding.severity == "high_risk" for finding in findings):
        return "HIGH_RISK"

    if any(finding.severity == "caution" for finding in findings):
        return "CAUTION"

    return "READY"


def assess_readiness(
    snapshot: DatasetSnapshot,
    frame: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
    leakage: Any = None,
) -> ReadinessReport:
    """Оценить готовность тройки «данные + задача + протокол».

    `leakage` — отчёт Leakage Guard, если он посчитан. Его находки входят в готовность
    напрямую: состояние «READY при подтверждённой блокирующей утечке» невозможно
    по построению, а не по договорённости (D-23).
    """
    from backend.readiness.checks import features, protocol_fit, size, target

    context = build_context(frame, spec, protocol)
    findings: list[Finding] = []

    for check in (
        lambda: size.check(frame, spec, context),
        lambda: target.check(frame, spec, protocol, context),
        lambda: features.check(snapshot, frame, spec, context),
        lambda: protocol_fit.check(frame, spec, protocol, context),
    ):
        findings.extend(check())

    if leakage is not None:
        findings.extend(_leakage_findings(leakage))

    order = {"high_risk": 0, "caution": 1, "info": 2}
    findings.sort(key=lambda finding: (not finding.blocking, order[finding.severity], finding.code))
    status = resolve_status(findings)

    return ReadinessReport(
        status=status,
        summary=_summarize(status, findings, context),
        context=context,
        findings=findings,
    )


def _leakage_findings(leakage: Any) -> list[Finding]:
    """Перевести сигналы утечки в находки готовности.

    Уровень доказательности сохраняется: подозрение по имени колонки не имеет права
    поднять статус готовности, а доказанный механизм — обязан.
    """
    severity_by_level = {"structural": "high_risk", "evidence": "high_risk", "heuristic": "info"}

    return [
        Finding(
            code=f"leakage.{signal.code}",
            severity=severity_by_level[signal.evidence_level],  # type: ignore[arg-type]
            scope=signal.scope,
            observed_value=signal.observed_value,
            threshold=signal.threshold,
            explanation=signal.explanation,
            consequence=signal.mechanism,
            suggested_action=signal.suggested_action,
            blocking=signal.blocking,
        )
        for signal in leakage.signals
    ]


def _summarize(status: Status, findings: list[Finding], context: ReadinessContext) -> str:
    if status == "READY":
        return (
            f"{context.usable_rows} строк с известной целью и {context.effective_dimension} "
            f"признаков после кодирования — примерно {context.rows_per_dimension} наблюдений "
            "на признак. Препятствий для честного сравнения не обнаружено."
        )

    blocking = [finding for finding in findings if finding.blocking]

    if blocking:
        return (
            "Честное сравнение невозможно: "
            + "; ".join(finding.explanation for finding in blocking[:2])
            + ". Пока это не исправлено, любые метрики будут вводить в заблуждение."
        )

    high = sum(finding.severity == "high_risk" for finding in findings)
    caution = sum(finding.severity == "caution" for finding in findings)

    return (
        f"Сравнение возможно, но результат нужно читать с оговорками: "
        f"{high} серьёзных замечаний и {caution} требующих внимания. "
        "Каждое ниже названо вместе с тем, что именно исказится."
    )
