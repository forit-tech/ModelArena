"""Управление прогонами: запуск, опрос, остановка.

Здесь живёт защита от случайного двойного запуска. Она **серверная**: блокировка кнопки
в интерфейсе не защищает ни от повторной отправки формы, ни от второй вкладки, ни от
клиента, который вообще не наш. Два одинаковых прогона — это не только вдвое больше
работы, но и две карточки одного эксперимента, между которыми потом никто не разберётся.

Прогон выполняется в фоновом потоке, а обучение — в отдельных процессах. Поток нужен,
чтобы HTTP-запрос не держал соединение до конца турнира; процессы — чтобы отмена была
настоящей, а падение чужой библиотеки не уносило backend.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from backend.arena.engine import ArenaEngine, write_folds
from backend.arena.events import ArenaEvent, EventLog, Progress, now
from backend.arena.spec import (
    DEFAULT_CONTENDER_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    RunSpec,
    build_run_spec,
)
from backend.arena.store import ContenderRecord, RunRecord, RunStore, new_run_id
from backend.core.errors import NotFoundError, ValidationError
from backend.models.context import ResourceBudget
from backend.models.registry import Contender
from backend.protocol.folds import FoldPlan
from backend.protocol.recommend import ProtocolProposal
from backend.tasks.spec import TaskSpec

logger = logging.getLogger("modelarena.arena")

POSITION_COLUMN = "__position__"


@dataclass
class _Active:
    engine: ArenaEngine
    log: EventLog
    thread: threading.Thread
    spec: RunSpec


def usable_row_ids(frame: pl.DataFrame, target_column: str) -> np.ndarray:
    """Позиции строк с известной целью в снимке датасета.

    Фолды строятся по отфильтрованной таблице, а предсказания должны ссылаться на строки
    снимка. Без этого соответствия join предсказаний с исходными данными съехал бы ровно
    на число строк с пустой целью — тихо и правдоподобно.
    """
    return (
        frame.with_row_index(name=POSITION_COLUMN)
        .filter(pl.col(target_column).is_not_null())[POSITION_COLUMN]
        .to_numpy()
        .astype(np.int64)
    )


class ArenaRunner:
    """Реестр активных и завершённых прогонов."""

    def __init__(self, store: RunStore | None = None) -> None:
        self._store = store or RunStore()
        self._active: dict[str, _Active] = {}
        self._lock = threading.Lock()
        recovered = self._store.recover_interrupted()

        if recovered:
            logger.warning(
                "Прогоны, не пережившие перезапуск backend, переведены в INTERRUPTED: %s",
                ", ".join(recovered),
            )

    @property
    def store(self) -> RunStore:
        return self._store

    # ------------------------------------------------------------------ запуск

    def start(
        self,
        *,
        dataset_id: str,
        dataset_fingerprint: str,
        dataset_path: Path,
        frame: pl.DataFrame,
        task: TaskSpec,
        protocol: ProtocolProposal,
        folds: FoldPlan,
        contenders: list[Contender],
        budget: ResourceBudget,
        analysis: dict[str, Any] | None = None,
        contender_timeout_seconds: int = DEFAULT_CONTENDER_TIMEOUT_SECONDS,
        run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS,
    ) -> tuple[RunRecord, bool]:
        """Запустить прогон. Второе значение — «это уже идущий прогон, а не новый».

        Совпадение отпечатка эксперимента с **активным** прогоном означает повторную
        отправку: пользователь получает тот же самый прогон, а не второй такой же.
        Завершённый прогон с тем же отпечатком повторить можно — это осознанное
        воспроизведение, а не случайность.
        """
        spec = build_run_spec(
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            task=task,
            protocol=protocol,
            folds=folds,
            contenders=contenders,
            budget=budget,
            contender_timeout_seconds=contender_timeout_seconds,
            run_timeout_seconds=run_timeout_seconds,
        )

        if not spec.runnable_contenders:
            raise ValidationError(
                "Ни один контендер не может быть запущен: причины перечислены в их статусах."
            )

        if spec.baseline_key is None:
            raise ValidationError(
                "В составе нет baseline. Без точки отсчёта результат сравнения не читается."
            )

        with self._lock:
            existing = self._store.find_active_by_fingerprint(spec.experiment_fingerprint)

            if existing is not None:
                return existing, True

            run_id = new_run_id()
            record = self._store.create(run_id, spec, _initial_records(contenders))

        directory = self._store.directory(run_id)
        write_folds(self._store.folds_path(run_id), folds, usable_row_ids(frame, task.target_column))

        if analysis is not None:
            #готовность и утечки на момент запуска сохраняются рядом с прогоном:
            #через месяц вопрос «а что тогда говорил Leakage Guard» возникает обязательно
            (directory / "analysis.json").write_text(
                json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )

        log = EventLog(path=self._store.events_path(run_id))
        log.append(
            ArenaEvent(
                kind="run_queued",
                at=now(),
                message=(
                    f"Прогон поставлен в очередь: {len(spec.runnable_contenders)} участников, "
                    f"{spec.n_splits} фолдов, разбиение {spec.fold_assignment_hash[:12]}."
                ),
            )
        )
        engine = ArenaEngine(
            record=record, spec=spec, store=self._store, dataset_path=dataset_path, log=log
        )
        thread = threading.Thread(
            target=self._execute, args=(run_id, engine), name=f"arena-{run_id}", daemon=True
        )

        with self._lock:
            self._active[run_id] = _Active(engine=engine, log=log, thread=thread, spec=spec)

        thread.start()
        return record, False

    def _execute(self, run_id: str, engine: ArenaEngine) -> None:
        try:
            engine.execute()
        except BaseException:
            #поток прогона не должен умирать молча: без записи состояние осталось бы
            #RUNNING навсегда, а причина не попала бы никуда
            logger.exception("Прогон %s прервался неожиданной ошибкой", run_id)

            try:
                record = self._store.get(run_id)
                record.state = "FAILED"
                record.error_code = "training_failed"
                record.error_message = "Прогон прервался внутренней ошибкой. Подробности в логе."
                self._store.save(record)
            except (OSError, NotFoundError, ValueError):
                logger.exception("Состояние прогона %s не удалось записать", run_id)

            raise
        finally:
            with self._lock:
                self._active.pop(run_id, None)

    # ------------------------------------------------------------------ чтение и отмена

    def cancel(self, run_id: str) -> RunRecord:
        with self._lock:
            active = self._active.get(run_id)

        if active is None:
            record = self._store.get(run_id)

            if record.state in {"PENDING", "RUNNING"}:
                #запись говорит RUNNING, но выполнять её некому: так выглядит прогон,
                #переживший перезапуск. Переводим в терминальное состояние, а не молчим
                self._store.recover_interrupted()
                return self._store.get(run_id)

            raise ValidationError(
                f"Прогон {run_id} уже завершён со статусом {record.state} и остановке не подлежит."
            )

        record = self._store.get(run_id)
        record.cancel_requested = True
        self._store.save(record)
        active.log.append(
            ArenaEvent(kind="cancel_requested", at=now(), message="Запрошена остановка прогона.")
        )
        active.engine.request_cancel()
        return record

    def get(self, run_id: str) -> RunRecord:
        return self._store.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return self._store.list_runs()

    def progress(self, run_id: str) -> Progress:
        with self._lock:
            active = self._active.get(run_id)

        if active is not None:
            return active.engine.progress()

        record = self._store.get(run_id)
        planned = [item for item in record.contenders if item.n_folds]
        return Progress(
            folds_completed=sum(item.folds_completed for item in planned),
            folds_planned=sum(item.n_folds for item in planned),
            contenders_finished=sum(
                1
                for item in record.contenders
                if item.state in {"SUCCEEDED", "FAILED", "CANCELLED"}
            ),
            contenders_planned=len(planned),
        )

    def events(self, run_id: str, since: int = 0) -> list[ArenaEvent]:
        with self._lock:
            active = self._active.get(run_id)

        if active is not None:
            return active.log.snapshot(since)

        #прогон уже не в памяти: журнал читается с диска, чтобы история осталась
        #доступной после перезапуска
        return EventLog.load(self._store.events_path(run_id)).snapshot(since)

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._active

    def wait(self, run_id: str, timeout: float | None = None) -> None:
        """Дождаться завершения прогона. Нужно тестам и корректному выключению."""
        with self._lock:
            active = self._active.get(run_id)

        if active is not None:
            active.thread.join(timeout)


def _initial_records(contenders: list[Contender]) -> list[ContenderRecord]:
    return [
        ContenderRecord(
            contender_key=contender.contender_key,
            label=contender.label,
            family=contender.family,
            preprocessing_profile=contender.preprocessing_profile,
            is_baseline=contender.is_baseline,
            selection_reason=contender.reason,
        )
        for contender in contenders
    ]


@lru_cache(maxsize=1)
def get_runner() -> ArenaRunner:
    return ArenaRunner()
