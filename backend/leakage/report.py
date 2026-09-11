"""Leakage Guard: ищет механизмы, из-за которых leaderboard окажется нечестным.

Утечка — свойство **связки** данных, задачи и протокола, а не колонки самой по себе (D-23).
Дубликат внутри обучающей части безвреден; он же, пересёкший границу train ↔ validation, —
конкретный механизм завышения. Поэтому проверки работают по фактическим индексам фолдов,
а не по наличию признака в датасете.

Три уровня доказательности, и они не взаимозаменяемы:

* `structural` — механизм доказан построением: значения совпадают с целью, дубликат
  фактически пересёк границу. Такое может блокировать;
* `evidence` — измерено **вне обучающей части** и механизм правдоподобен;
* `heuristic` — подозрение по форме: имя колонки. **Никогда не блокирует.**

Главное, чего здесь нет: правила «высокая корреляция значит утечка». Сильный честный
предиктор существует и встречается чаще, чем утечка. Детектор, кричащий на всё,
бесполезен — доверять можно только тому, чьё молчание тоже что-то значит.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import polars as pl

from backend.protocol.folds import FoldPlan
from backend.protocol.recommend import ProtocolProposal
from backend.tasks.spec import TaskSpec

EvidenceLevel = Literal["structural", "evidence", "heuristic"]
LeakageRisk = Literal["none", "suspected", "high", "confirmed"]

#доля объяснённой дисперсии, выше которой один признак в одиночку восстанавливает цель
#почти точно. Порог намеренно близок к единице: 0.95 — это часто просто хороший признак
SINGLE_FEATURE_REGRESSION = 0.999
SINGLE_FEATURE_CLASSIFICATION = 0.995
#абсолютное совпадение значений с точностью float
EXACT_MATCH_TOLERANCE = 1e-9
#разница долей положительного класса между «пропуск есть» и «пропуска нет»
MISSINGNESS_ASSOCIATION = 0.5
#минимум строк в каждой части, чтобы разница долей вообще что-то значила
MIN_ROWS_FOR_MISSINGNESS = 30
#слова, которые в имени часто означают событие ПОСЛЕ определения цели
POST_OUTCOME_HINTS = (
    "reason", "outcome", "result", "resolution", "closed", "cancel", "refund",
    "termination", "final", "post_", "_after", "причина", "итог", "результат",
    "закрыт", "отмен", "возврат",
)


@dataclass(frozen=True)
class LeakageSignal:
    code: str
    evidence_level: EvidenceLevel
    scope: str
    columns: list[str]
    observed_value: Any
    threshold: str
    explanation: str
    #чем именно этот сигнал делает сравнение моделей нечестным
    mechanism: str
    suggested_action: str
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NotEvaluated:
    """Проверка не выполнена. **Это не то же самое, что «утечки нет».**

    Разница принципиальна и уже проявилась на практике: расхождение «True» и «true»
    заставило проверку пропуска молчать не потому, что механизма не было, а потому что
    она сама сломалась семантически. Молчание сломанной проверки выглядело как чистый
    результат — ровно та ложная безопасность, которой быть не должно.
    """

    check: str
    scope: str
    reason: str
    consequence: str = (
        "По этому механизму вывод не сделан ни в одну сторону: отсутствие сигнала здесь "
        "ничего не доказывает."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeakageReport:
    risk: LeakageRisk
    summary: str
    signals: list[LeakageSignal] = field(default_factory=list)
    not_evaluated: list[NotEvaluated] = field(default_factory=list)
    #формулировка обязана оставаться осторожной: инструмент не может доказать отсутствие утечки
    disclaimer: str = (
        "Guard находит механизмы, которыми утечка обычно проявляется. Он не может доказать "
        "ни её наличие, ни отсутствие: знает, в какой момент реального процесса становится "
        "известно каждое поле, только человек."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "summary": self.summary,
            "disclaimer": self.disclaimer,
            "signals": [signal.to_dict() for signal in self.signals],
            "not_evaluated": [item.to_dict() for item in self.not_evaluated],
        }

    @property
    def fully_evaluated(self) -> bool:
        return not self.not_evaluated

    def by_code(self, code: str) -> LeakageSignal | None:
        return next((signal for signal in self.signals if signal.code == code), None)

    @property
    def blocking_signals(self) -> list[LeakageSignal]:
        return [signal for signal in self.signals if signal.blocking]


def inspect_leakage(
    frame: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
    folds: FoldPlan,
) -> LeakageReport:
    """Проверить связку на механизмы утечки."""
    from backend.leakage.checks import boundaries, derived, naming, predictive

    usable = frame.filter(pl.col(spec.target_column).is_not_null())
    signals: list[LeakageSignal] = []
    skipped: list[NotEvaluated] = []

    for check in (
        lambda: derived.check(usable, spec),
        lambda: boundaries.check(usable, spec, protocol, folds),
        lambda: predictive.check(usable, spec, folds),
        lambda: naming.check(spec),
    ):
        found, not_done = _split_outcome(check())
        signals.extend(found)
        skipped.extend(not_done)

    order = {"structural": 0, "evidence": 1, "heuristic": 2}
    signals.sort(key=lambda signal: (not signal.blocking, order[signal.evidence_level], signal.code))

    return LeakageReport(
        risk=_resolve_risk(signals),
        summary=_summarize(signals, skipped),
        signals=signals,
        not_evaluated=skipped,
    )


def _split_outcome(
    outcome: list[Any],
) -> tuple[list[LeakageSignal], list[NotEvaluated]]:
    #проверки возвращают смесь: найденные сигналы и отметки о том, что выполнить не удалось
    return (
        [item for item in outcome if isinstance(item, LeakageSignal)],
        [item for item in outcome if isinstance(item, NotEvaluated)],
    )


def _resolve_risk(signals: list[LeakageSignal]) -> LeakageRisk:
    if any(signal.blocking for signal in signals):
        return "confirmed"

    #структурный и измеренный сигналы дают один уровень: оба означают конкретный механизм,
    #и разделять их на разные риски было бы разделением без разницы
    if any(signal.evidence_level in {"structural", "evidence"} for signal in signals):
        return "high"

    if signals:
        #одни только подозрения по имени не поднимают риск выше «стоит посмотреть»
        return "suspected"

    return "none"


def _summarize(signals: list[LeakageSignal], skipped: list[NotEvaluated]) -> str:
    incomplete = (
        f" При этом {len(skipped)} проверок выполнить не удалось — по ним вывода нет."
        if skipped
        else ""
    )

    if not signals:
        return (
            "Механизмов утечки не обнаружено: копий цели среди признаков нет, границы фолдов "
            "не пересекаются дубликатами и сущностями, отдельный признак цель не восстанавливает."
            + incomplete
        )

    blocking = [signal for signal in signals if signal.blocking]

    if blocking:
        return (
            "Сравнение моделей на этих данных бессмысленно: "
            + "; ".join(signal.explanation for signal in blocking[:2])
            + "."
        )

    structural = sum(signal.evidence_level == "structural" for signal in signals)
    evidence = sum(signal.evidence_level == "evidence" for signal in signals)
    heuristic = sum(signal.evidence_level == "heuristic" for signal in signals)

    parts = []

    if structural:
        parts.append(f"{structural} с доказанным механизмом")

    if evidence:
        parts.append(f"{evidence} подтверждённых измерением")

    if heuristic:
        parts.append(f"{heuristic} подозрений по форме")

    return "Найдено сигналов: " + ", ".join(parts) + ". Каждый ниже объясняет свой механизм." + incomplete
