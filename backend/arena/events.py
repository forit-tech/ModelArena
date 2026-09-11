"""Событийный прогресс.

Никаких «37%», выведенных из времени или из ощущений. Прогресс здесь — это счёт
**фактически случившихся** событий: столько-то фолдов посчитано из стольких-то
запланированных, столько-то контендеров завершено из стольких-то.

Внутренний прогресс обучения модель не отдаёт, и выдумывать его нельзя: полоса,
доросшая до 80% и стоящая там десять минут, хуже честного «фолд 2 из 5».
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

EventKind = Literal[
    "run_queued",
    "run_started",
    "contender_started",
    "fold_completed",
    "contender_finished",
    "contender_skipped",
    "cancel_requested",
    "run_finished",
]


@dataclass(frozen=True)
class ArenaEvent:
    kind: EventKind
    at: str
    message: str
    contender_key: str | None = None
    fold: int | None = None
    n_folds: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class EventLog:
    """Дописываемый журнал событий прогона.

    Хранится в памяти для быстрых опросов и дублируется в `events.jsonl`, чтобы
    после перезапуска backend было видно, до какого места дошёл прерванный прогон.
    Запись построчная и дописывающая: обрыв на середине портит последнюю строку,
    а не весь журнал.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._events: list[ArenaEvent] = []
        self._lock = threading.Lock()

    def append(self, event: ArenaEvent) -> None:
        with self._lock:
            self._events.append(event)

            if self._path is None:
                return

            try:
                with self._path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            except OSError:
                #журнал — вспомогательный артефакт: невозможность его дописать не должна
                #ронять прогон, но и молчать об этом нельзя
                self._events.append(
                    ArenaEvent(
                        kind=event.kind,
                        at=now(),
                        message="Журнал событий не записывается на диск: нет доступа к файлу.",
                    )
                )

    def snapshot(self, since: int = 0) -> list[ArenaEvent]:
        with self._lock:
            return list(self._events[since:])

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    @classmethod
    def load(cls, path: Path) -> EventLog:
        log = cls(path=None)

        if not path.exists():
            return log

        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                #последняя строка могла оборваться на середине записи
                continue

            known = set(ArenaEvent.__dataclass_fields__)
            log._events.append(ArenaEvent(**{key: value for key, value in payload.items() if key in known}))

        return log


@dataclass(frozen=True)
class Progress:
    """Прогресс, собранный только из завершённых единиц работы."""

    folds_completed: int
    folds_planned: int
    contenders_finished: int
    contenders_planned: int

    def to_dict(self) -> dict[str, Any]:
        #процент считается по фолдам — единственной единице, факт завершения которой
        #система реально наблюдает. Если планировать нечего, доли нет вовсе:
        #ноль в знаменателе не заменяется на «0%», это разные утверждения
        return {
            "folds_completed": self.folds_completed,
            "folds_planned": self.folds_planned,
            "contenders_finished": self.contenders_finished,
            "contenders_planned": self.contenders_planned,
            "completed_fraction": (
                round(self.folds_completed / self.folds_planned, 4) if self.folds_planned else None
            ),
        }
