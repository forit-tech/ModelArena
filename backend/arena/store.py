"""Хранение прогонов и восстановление после перезапуска.

Два свойства, ради которых слой существует отдельно от движка.

**Атомарная публикация.** Результат контендера собирается в staging-каталоге и переносится
на место одним движением. Обучение, запись предсказаний и запись метаданных — три
операции, и падение между ними не имеет права оставить «успешный» полуартефакт: карточка
с метриками, но без предсказаний, выглядит достоверно и не проверяется ничем.

**Честный итог после перезапуска.** Прогон, помеченный RUNNING, после смерти backend
остаётся RUNNING навсегда, если этого не разобрать. Такой прогон висит в списке вечно
и обещает результат, который никто не считает. Поэтому у записи есть отметка процесса-
владельца, и чужой незавершённый прогон переводится в терминальное `INTERRUPTED`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.arena.spec import RunSpec
from backend.arena.states import ContenderState, RunState
from backend.core.config import get_settings
from backend.core.errors import NotFoundError, ValidationError

logger = logging.getLogger("modelarena.arena")

RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"
FOLDS_FILE = "folds.npz"
CONTENDERS_DIR = "contenders"
STAGING_DIR = ".staging"
RUN_ID_PATTERN = re.compile(r"^run_[0-9a-f]{16,48}$")

#метка текущего процесса backend. Прогон, помеченный чужой меткой и не завершённый,
#пережил перезапуск и продолжаться уже не будет
OWNER_TOKEN = secrets.token_hex(8)


def new_run_id() -> str:
    return f"run_{int(time.time() * 1000):011x}{secrets.token_hex(5)}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ContenderRecord:
    """Состояние одного участника. Причина обязательна для всего, кроме успеха."""

    contender_key: str
    label: str
    family: str
    preprocessing_profile: str
    is_baseline: bool
    state: ContenderState = "PENDING"
    #причина статуса до старта (недоступен, неуместен) — приходит из реестра адаптеров
    selection_reason: str = ""
    error_code: str | None = None
    error_message: str = ""
    folds_completed: int = 0
    n_folds: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    coverage: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    state: RunState
    spec: dict[str, Any]
    contenders: list[ContenderRecord]
    owner_token: str = OWNER_TOKEN
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    error_code: str | None = None
    error_message: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def experiment_fingerprint(self) -> str:
        return str(self.spec.get("experiment_fingerprint", ""))

    def contender(self, key: str) -> ContenderRecord:
        for record in self.contenders:
            if record.contender_key == key:
                return record

        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "state": self.state,
            "cancel_requested": self.cancel_requested,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "notes": self.notes,
            "owner_token": self.owner_token,
            "spec": self.spec,
            "contenders": [record.to_dict() for record in self.contenders],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunRecord:
        known = set(ContenderRecord.__dataclass_fields__)
        contenders = [
            ContenderRecord(**{key: value for key, value in item.items() if key in known})
            for item in payload.get("contenders", [])
        ]
        return cls(
            run_id=payload["run_id"],
            created_at=payload["created_at"],
            state=payload["state"],
            spec=payload.get("spec", {}),
            contenders=contenders,
            owner_token=payload.get("owner_token", ""),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            cancel_requested=bool(payload.get("cancel_requested", False)),
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message", ""),
            notes=list(payload.get("notes", [])),
        )


class RunStore:
    """Файловое хранилище прогонов."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or get_settings().runs_directory
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def directory(self, run_id: str) -> Path:
        #идентификатор попадает в путь, поэтому проверяется по белому списку символов
        if not RUN_ID_PATTERN.match(run_id):
            raise ValidationError(f"Некорректный идентификатор прогона: {run_id}")

        return self._root / run_id

    # ------------------------------------------------------------------ запись

    def create(self, run_id: str, spec: RunSpec, contenders: list[ContenderRecord]) -> RunRecord:
        directory = self.directory(run_id)
        (directory / CONTENDERS_DIR).mkdir(parents=True, exist_ok=True)
        (directory / STAGING_DIR).mkdir(parents=True, exist_ok=True)

        record = RunRecord(
            run_id=run_id,
            created_at=_utc_now(),
            state="PENDING",
            spec=spec.to_dict(),
            contenders=contenders,
        )
        self.save(record)
        return record

    def save(self, record: RunRecord) -> None:
        """Записать карточку прогона целиком и атомарно.

        Дописывание по месту оставило бы при обрыве полуразобранный JSON, и прогон
        стал бы нечитаемым — вместе со ссылками на уже посчитанные предсказания.
        """
        directory = self.directory(record.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".{RUN_FILE}.{os.getpid()}.tmp"

        with self._lock:
            temporary.write_text(
                json.dumps(record.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, directory / RUN_FILE)

    def staging_for(self, run_id: str, contender_key: str) -> Path:
        path = self.directory(run_id) / STAGING_DIR / _safe_key(contender_key)

        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

        path.mkdir(parents=True, exist_ok=True)
        return path

    def publish(self, run_id: str, contender_key: str) -> Path:
        """Перенести готовый результат контендера из staging на место.

        Переносится **каталог целиком**: предсказания и метаданные становятся видимы
        одновременно. Появление метаданных без предсказаний означало бы карточку
        с метриками, которую нечем перепроверить.
        """
        staging = self.directory(run_id) / STAGING_DIR / _safe_key(contender_key)
        final = self.directory(run_id) / CONTENDERS_DIR / _safe_key(contender_key)

        if not staging.exists():
            raise NotFoundError(f"Результат контендера {contender_key} не найден в staging.")

        if final.exists():
            shutil.rmtree(final, ignore_errors=True)

        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        return final

    def discard_staging(self, run_id: str, contender_key: str) -> None:
        shutil.rmtree(
            self.directory(run_id) / STAGING_DIR / _safe_key(contender_key), ignore_errors=True
        )

    # ------------------------------------------------------------------ чтение

    def events_path(self, run_id: str) -> Path:
        return self.directory(run_id) / EVENTS_FILE

    def folds_path(self, run_id: str) -> Path:
        return self.directory(run_id) / FOLDS_FILE

    def predictions_path(self, run_id: str, contender_key: str) -> Path:
        from backend.arena.worker import PREDICTIONS_FILE

        return self.directory(run_id) / CONTENDERS_DIR / _safe_key(contender_key) / PREDICTIONS_FILE

    def get(self, run_id: str) -> RunRecord:
        path = self.directory(run_id) / RUN_FILE

        if not path.exists():
            raise NotFoundError(f"Прогон {run_id} не найден.")

        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(self) -> list[RunRecord]:
        if not self._root.exists():
            return []

        records: list[RunRecord] = []

        for directory in self._root.iterdir():
            path = directory / RUN_FILE

            if not directory.is_dir() or not path.exists():
                continue

            try:
                records.append(RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                #повреждённая карточка не должна ронять список, но и исчезать молча
                #не имеет права: прогон остаётся на диске и не находится по отпечатку
                logger.warning("Карточка прогона %s не читается: %s", directory.name, error)
                continue

        records.sort(key=lambda record: record.created_at, reverse=True)
        return records

    # ------------------------------------------------------------------ восстановление

    def recover_interrupted(self) -> list[str]:
        """Перевести прогоны, пережившие перезапуск backend, в терминальное состояние.

        Продолжить их нельзя: процессы обучения умерли вместе с backend. Оставить
        как есть — значит держать в списке вечное RUNNING, за которым ничего не идёт.
        Уже завершённые контендеры и их предсказания при этом сохраняются: они посчитаны
        честно, и терять их из-за перезапуска незачем.
        """
        recovered: list[str] = []

        for record in self.list_runs():
            if record.state not in {"PENDING", "RUNNING"} or record.owner_token == OWNER_TOKEN:
                continue

            for contender in record.contenders:
                if contender.state in {"PENDING", "RUNNING"}:
                    contender.state = "CANCELLED"
                    contender.error_code = "interrupted"
                    contender.error_message = (
                        "Backend был перезапущен во время обучения; контендер не завершился."
                    )
                    contender.finished_at = _utc_now()

            succeeded = [item for item in record.contenders if item.state == "SUCCEEDED"]
            record.state = "INTERRUPTED"
            record.error_code = "interrupted"
            record.error_message = (
                f"Backend был перезапущен во время прогона. Завершённых контендеров: "
                f"{len(succeeded)} из {len(record.contenders)}; их результаты сохранены."
            )
            record.finished_at = _utc_now()
            record.owner_token = OWNER_TOKEN
            self.save(record)
            recovered.append(record.run_id)
            #staging прерванных контендеров не публикуется: там может лежать результат,
            #записанный наполовину
            shutil.rmtree(self.directory(record.run_id) / STAGING_DIR, ignore_errors=True)
            (self.directory(record.run_id) / STAGING_DIR).mkdir(parents=True, exist_ok=True)

        return recovered

    def find_active_by_fingerprint(self, fingerprint: str) -> RunRecord | None:
        return next(
            (
                record
                for record in self.list_runs()
                if record.experiment_fingerprint == fingerprint
                and record.state in {"PENDING", "RUNNING"}
            ),
            None,
        )


def _safe_key(contender_key: str) -> str:
    #ключ адаптера попадает в путь: он приходит из нашего же реестра, но проверка стоит
    #здесь, а не в вызывающем коде, чтобы новый адаптер с точкой в имени не создал
    #каталог за пределами прогона
    if not re.match(r"^[a-z0-9_]{1,64}$", contender_key):
        raise ValidationError(f"Недопустимый ключ контендера: {contender_key}")

    return contender_key
