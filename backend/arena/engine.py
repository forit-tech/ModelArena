"""Планировщик прогона: кто, когда и с какими ресурсами обучается.

Движок держит три обещания, каждое из которых легко нарушить незаметно.

**`FoldPlan` — закон.** Здесь нет ни одного конструктора разбиения. Индексы приходят
готовыми и передаются как есть. Стоит движку построить свой `KFold` «для удобства» —
и вся работа предыдущих этапов (группы, время, скользящее окно, `assignment_hash`,
границы утечки) обесценивается, а метрика при этом останется правдоподобной.

**Падение одного не уносит остальных.** Каждый контендер живёт в своём процессе,
и его авария — отказ участника, а не прогона.

**Отмена настоящая.** Она не переключает состояние в интерфейсе, а перестаёт запускать
новых участников и завершает процессы уже запущенных. Библиотеку, которая считает
внутри нативного кода, иначе остановить нельзя: флаг она не прочитает.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.arena.events import ArenaEvent, EventLog, Progress, now
from backend.arena.spec import RunSpec
from backend.arena.states import resolve_run_state
from backend.arena.store import ContenderRecord, RunRecord, RunStore
from backend.arena.worker import RESULT_FILE, run_contender_process
from backend.protocol.folds import FoldPlan

logger = logging.getLogger("modelarena.arena")

#сколько ждать завершения процесса по-хорошему, прежде чем убивать. Нативный код
#может не отреагировать на мягкое завершение вовсе
TERMINATE_GRACE_SECONDS = 5.0
KILL_GRACE_SECONDS = 2.0
POLL_INTERVAL_SECONDS = 0.1
#сколько ждать процесс, который уже записал результат, но не закончился сам.
#Нативная библиотека не обязана сворачиваться быстро, а иногда не сворачивается вовсе
EXIT_GRACE_SECONDS = 10.0


def write_folds(path: Path, folds: FoldPlan, row_ids: np.ndarray) -> None:
    """Сохранить утверждённое разбиение для дочерних процессов.

    Индексы передаются файлом, а не через pickle аргументов: при способе запуска `spawn`
    распаковка аргументов тянет numpy раньше, чем процесс успевает ограничить число
    потоков BLAS.

    `row_ids` — соответствие «позиция среди строк с известной целью» → «строка снимка».
    Без него предсказания были бы подписаны позициями отфильтрованной таблицы, и join
    с исходными данными молча съехал бы на число выброшенных строк.
    """
    arrays: dict[str, np.ndarray] = {
        "holdout": folds.holdout.astype(np.int64),
        "train_pool": folds.train_pool.astype(np.int64),
        "row_ids": row_ids.astype(np.int64),
        "n_folds": np.array(folds.n_splits, dtype=np.int64),
    }

    for index, (training, validation) in enumerate(folds.folds):
        arrays[f"train_{index}"] = training.astype(np.int64)
        arrays[f"val_{index}"] = validation.astype(np.int64)

    np.savez(path, **arrays)


@dataclass
class _Worker:
    key: str
    process: Any
    staging: Path
    started_at: float
    #момент, когда на диске появился результат: с него начинается отсчёт терпения
    #к процессу, который сделал работу, но не завершился
    result_seen_at: float | None = None
    forced_exit: bool = False


class ArenaEngine:
    """Выполняет один прогон. Экземпляр одноразовый."""

    def __init__(
        self,
        *,
        record: RunRecord,
        spec: RunSpec,
        store: RunStore,
        dataset_path: Path,
        log: EventLog,
    ) -> None:
        self._record = record
        self._spec = spec
        self._store = store
        self._dataset_path = dataset_path
        self._log = log
        #две разные причины перестать запускать новых участников: остановка пользователем
        #и исчерпанный предел времени на прогон. Итоговое состояние у них разное,
        #и один общий флаг назвал бы упёршийся в лимит прогон отменённым
        self._cancel = threading.Event()
        self._stop_scheduling = threading.Event()
        self._context = mp.get_context("spawn")
        self._events: Any = self._context.Queue()
        self._started = time.monotonic()

    # ------------------------------------------------------------------ управление

    def request_cancel(self) -> None:
        """Пометить прогон к остановке. Реальное завершение процессов делает цикл."""
        self._cancel.set()
        self._stop_scheduling.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def progress(self) -> Progress:
        planned = [item for item in self._record.contenders if item.n_folds]
        return Progress(
            folds_completed=sum(item.folds_completed for item in planned),
            folds_planned=sum(item.n_folds for item in planned),
            contenders_finished=sum(
                1
                for item in self._record.contenders
                if item.state in {"SUCCEEDED", "FAILED", "CANCELLED"}
            ),
            contenders_planned=len(planned),
        )

    # ------------------------------------------------------------------ выполнение

    def execute(self) -> RunRecord:
        self._record.state = "RUNNING"
        self._record.started_at = _utc_now()
        self._save()
        self._emit(ArenaEvent(kind="run_started", at=now(), message="Прогон начался."))

        self._mark_skipped()
        pending = [
            item
            for item in self._record.contenders
            if item.state == "PENDING" and self._is_runnable(item.contender_key)
        ]
        running: dict[str, _Worker] = {}

        try:
            self._loop(pending, running)
        finally:
            #что бы ни случилось выше, ни один дочерний процесс не остаётся жить:
            #осиротевший worker продолжает занимать ядра и память уже после того,
            #как прогон исчез из списка активных
            for worker in list(running.values()):
                self._stop_worker(worker, state="CANCELLED", code="cancelled")

            self._drain_queue()

        self._finish()
        return self._record

    def _loop(self, pending: list[ContenderRecord], running: dict[str, _Worker]) -> None:
        while pending or running:
            if self._cancel.is_set():
                #отмена не ждёт, пока запущенные досчитают: их процессы завершаются
                #здесь же. Ожидание естественного конца было бы отменой только на экране
                self._abort(pending, running)
                return

            while (
                not self._stop_scheduling.is_set()
                and pending
                and len(running) < self._spec.budget.max_parallel_fits
            ):
                record = pending.pop(0)
                running[record.contender_key] = self._start(record)

            if self._stop_scheduling.is_set() and pending:
                #в очередь больше никто не пойдёт: оставшиеся получают терминальное
                #состояние с причиной, а не висят PENDING навсегда
                for record in pending:
                    self._close(record, state="CANCELLED", code="cancelled")

                pending.clear()

            self._drain_queue()
            self._reap(running)
            self._enforce_limits(running)

            if not running and not pending:
                return

            time.sleep(POLL_INTERVAL_SECONDS)

    def _abort(self, pending: list[ContenderRecord], running: dict[str, _Worker]) -> None:
        for record in pending:
            self._close(record, state="CANCELLED", code="cancelled")

        pending.clear()

        for key, worker in list(running.items()):
            running.pop(key, None)
            self._stop_worker(worker, state="CANCELLED", code="cancelled")

    # ------------------------------------------------------------------ шаги

    def _mark_skipped(self) -> None:
        """Отметить участников, не запускаемых по составу, до старта.

        Недоступная библиотека и неуместная на этих данных модель — это **не отказ
        обучения**. Они определены заранее, с причиной, и в знаменатель «упавших»
        не входят: иначе прогон с четырьмя моделями из десяти выглядел бы провальным,
        хотя шесть остальных никто и не собирался запускать.
        """
        by_key = {item.contender_key: item for item in self._spec.contenders}

        for record in self._record.contenders:
            contender = by_key.get(record.contender_key)

            if contender is None or contender.runnable:
                continue

            record.state = "SKIPPED"
            record.error_code = (
                "contender_unavailable"
                if contender.status == "unavailable"
                else "incompatible_task"
            )
            record.error_message = contender.reason
            record.finished_at = _utc_now()
            self._emit(
                ArenaEvent(
                    kind="contender_skipped",
                    at=now(),
                    message=f"{record.label}: {contender.reason}",
                    contender_key=record.contender_key,
                )
            )

        self._save()

    def _is_runnable(self, key: str) -> bool:
        return any(
            item.contender_key == key and item.runnable for item in self._spec.contenders
        )

    def _start(self, record: ContenderRecord) -> _Worker:
        contender = next(
            item for item in self._spec.contenders if item.contender_key == record.contender_key
        )
        staging = self._store.staging_for(self._record.run_id, record.contender_key)
        payload = {
            "contender_key": contender.contender_key,
            "adapter_key": contender.adapter_key,
            "params": contender.params,
            "preprocessing_profile": contender.preprocessing_profile,
            "supports_proba": contender.supports_proba,
            "task_type": self._spec.task.task_type,
            "class_labels": list(self._spec.task.class_labels),
            "target_column": self._spec.task.target_column,
            "feature_columns": list(self._spec.task.feature_columns),
            "strategy": self._spec.protocol.cv_splitter,
            "seed": self._spec.seed,
            "budget": {
                "max_parallel_fits": self._spec.budget.max_parallel_fits,
                "threads_per_fit": self._spec.budget.threads_per_fit,
                "memory_budget_mb": self._spec.budget.memory_budget_mb,
            },
            #потоки берутся из бюджета и ставятся в дочернем процессе до импорта numpy:
            #два контендера по семь потоков должны дать четырнадцать рабочих потоков,
            #а не два захвата всех ядер
            "threads_per_fit": self._spec.budget.threads_per_fit,
            "dataset_path": str(self._dataset_path),
            "folds_path": str(self._store.folds_path(self._record.run_id)),
            "staging_directory": str(staging),
        }
        process = self._context.Process(
            target=run_contender_process, args=(payload, self._events), daemon=False
        )
        process.start()

        record.state = "RUNNING"
        record.started_at = _utc_now()
        record.n_folds = self._spec.n_splits
        self._save()
        self._emit(
            ArenaEvent(
                kind="contender_started",
                at=now(),
                message=f"{record.label}: обучение началось.",
                contender_key=record.contender_key,
                n_folds=self._spec.n_splits,
            )
        )
        return _Worker(
            key=record.contender_key,
            process=process,
            staging=staging,
            started_at=time.monotonic(),
        )

    def _drain_queue(self) -> None:
        while True:
            try:
                message = self._events.get(timeout=POLL_INTERVAL_SECONDS)
            except (queue.Empty, OSError, ValueError):
                return

            if message.get("kind") != "fold_completed":
                continue

            key = message.get("contender_key", "")

            try:
                record = self._record.contender(key)
            except KeyError:
                continue

            record.folds_completed = int(message.get("fold", 0)) + 1
            self._emit(
                ArenaEvent(
                    kind="fold_completed",
                    at=now(),
                    message=(
                        f"{record.label}: фолд {record.folds_completed} из "
                        f"{message.get('n_folds', record.n_folds)} посчитан."
                    ),
                    contender_key=key,
                    fold=record.folds_completed,
                    n_folds=int(message.get("n_folds", record.n_folds)),
                    payload={"seconds": message.get("seconds")},
                )
            )

    def _reap(self, running: dict[str, _Worker]) -> None:
        """Забрать результаты завершившихся и не ждать вечно тех, кто уже всё сделал.

        Результат на диске — единственный источник истины о том, что контендер посчитан.
        Процесс, который его записал, но продолжает жить, работу уже сделал: чаще всего
        внутри него остался пул потоков сторонней библиотеки. Ждать его бесконечно значит
        держать прогон в RUNNING из-за чужого сворачивания, поэтому после отсрочки он
        завершается принудительно, а результат забирается как обычно.
        """
        for key, worker in list(running.items()):
            if not worker.process.is_alive():
                running.pop(key, None)
                self._collect(worker)
                continue

            if not (worker.staging / RESULT_FILE).exists():
                continue

            if worker.result_seen_at is None:
                worker.result_seen_at = time.monotonic()
                continue

            if time.monotonic() - worker.result_seen_at < EXIT_GRACE_SECONDS:
                continue

            logger.warning(
                "Контендер %s записал результат, но процесс не завершился за %.0f с — завершаем",
                key,
                EXIT_GRACE_SECONDS,
            )
            worker.forced_exit = True
            worker.process.terminate()
            worker.process.join(TERMINATE_GRACE_SECONDS)

            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join(KILL_GRACE_SECONDS)

            running.pop(key, None)
            self._collect(worker)

    def _enforce_limits(self, running: dict[str, _Worker]) -> None:
        elapsed_run = time.monotonic() - self._started

        for key, worker in list(running.items()):
            elapsed = time.monotonic() - worker.started_at

            if elapsed > self._spec.contender_timeout_seconds:
                running.pop(key, None)
                self._stop_worker(worker, state="FAILED", code="timeout")

        if elapsed_run > self._spec.run_timeout_seconds and running:
            #предел на прогон целиком: несколько контендеров, каждый в рамках своего
            #лимита, всё равно способны держать машину сутки
            for key, worker in list(running.items()):
                running.pop(key, None)
                self._stop_worker(worker, state="FAILED", code="run_timeout")

            self._record.error_code = "run_timeout"
            self._record.error_message = (
                f"Прогон остановлен по пределу времени ({self._spec.run_timeout_seconds} с)."
            )
            #упёршийся в лимит прогон — не отменённый: новых не запускаем, но итог
            #считается по фактическим результатам, а не объявляется отменой
            self._stop_scheduling.set()

    def _stop_worker(self, worker: _Worker, *, state: str, code: str) -> None:
        """Завершить процесс контендера и записать честную причину.

        Мягкое завершение может не сработать: нативный код не обязан его замечать.
        Поэтому после ожидания процесс убивается — иначе «остановленный» прогон
        продолжал бы молотить процессор.
        """
        process = worker.process

        if process.is_alive():
            process.terminate()
            process.join(TERMINATE_GRACE_SECONDS)

        if process.is_alive():
            process.kill()
            process.join(KILL_GRACE_SECONDS)

        self._store.discard_staging(self._record.run_id, worker.key)

        try:
            record = self._record.contender(worker.key)
        except KeyError:
            return

        if record.state in {"SUCCEEDED", "FAILED", "CANCELLED", "SKIPPED"}:
            return

        self._close(record, state=state, code=code)

    def _collect(self, worker: _Worker) -> None:
        """Разобрать завершившийся процесс: результат или авария."""
        record = self._record.contender(worker.key)
        result_path = worker.staging / RESULT_FILE

        if not result_path.exists():
            #процесс закончился, не оставив результата: это авария нативного кода,
            #а не ошибка, которую он успел описать
            exit_code = worker.process.exitcode
            self._store.discard_staging(self._record.run_id, worker.key)
            self._close(
                record,
                state="FAILED",
                code="worker_crashed",
                message=(
                    f"Процесс контендера завершился с кодом {exit_code}, не оставив результата. "
                    "Подробности — в логе backend."
                ),
            )
            return

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        published = self._store.publish(self._record.run_id, worker.key)
        record.elapsed_seconds = payload.get("elapsed_seconds")
        record.warnings = list(payload.get("warnings", []))
        record.coverage = payload.get("coverage")

        if payload.get("status") != "SUCCEEDED":
            self._close(
                record,
                state="FAILED",
                code=payload.get("error_code") or "training_failed",
                message=payload.get("error_message", ""),
            )
            return

        record.folds_completed = max(record.folds_completed, len(payload.get("folds", [])))

        if worker.forced_exit:
            record.warnings = [
                *record.warnings,
                "Процесс контендера пришлось завершить принудительно: результат он записал, "
                "но сам не закончился. На сами предсказания это не влияет.",
            ]

        self._compute_metrics(record, published)

    def _compute_metrics(self, record: ContenderRecord, published: Path) -> None:
        from backend.arena.metrics import evaluate, read_predictions
        from backend.arena.worker import PREDICTIONS_FILE

        try:
            frame = read_predictions(published / PREDICTIONS_FILE)
            sets = evaluate(
                frame,
                task_type=self._spec.task.task_type,
                class_labels=list(self._spec.task.class_labels),
                positive_label=self._spec.task.positive_label,
            )
        except (OSError, ValueError) as error:
            #предсказания остаются на диске: они посчитаны честно и пригодятся.
            #Но без метрик участник не сравним, и место в leaderboard он занять не может
            logger.warning("Метрики для %s не посчитались: %s", record.contender_key, error)
            self._close(
                record,
                state="FAILED",
                code="metric_failed",
                message=(
                    "Предсказания сохранены, но метрики по ним посчитать не удалось: "
                    f"{type(error).__name__}."
                ),
            )
            return

        record.metrics = [item.to_dict() for item in sets]
        self._close(record, state="SUCCEEDED", code=None)

    def _close(
        self,
        record: ContenderRecord,
        *,
        state: str,
        code: str | None,
        message: str = "",
    ) -> None:
        from backend.arena.states import describe_error

        record.state = state  # type: ignore[assignment]
        record.error_code = code
        record.error_message = message or (describe_error(code) if code else "")
        record.finished_at = _utc_now()
        self._save()
        self._emit(
            ArenaEvent(
                kind="contender_finished",
                at=now(),
                message=f"{record.label}: {state}."
                + (f" {record.error_message}" if record.error_message else ""),
                contender_key=record.contender_key,
                payload={"state": state, "error_code": code},
            )
        )

    def _finish(self) -> None:
        baseline_key = self._spec.baseline_key
        baseline = next(
            (item for item in self._record.contenders if item.contender_key == baseline_key), None
        )
        baseline_failed = baseline is not None and baseline.state == "FAILED"

        self._record.state = resolve_run_state(
            contender_states=[item.state for item in self._record.contenders],
            cancelled=self._cancel.is_set() and not baseline_failed,
            baseline_failed=baseline_failed,
        )

        if baseline_failed:
            #baseline не обучается на признаках вовсе. Если не смог он, сломан общий путь
            #данных, а не одна модель, и остальные результаты нельзя считать осмысленными
            self._record.error_code = "baseline_failed"
            self._record.error_message = (
                f"Baseline «{baseline.label}» не обучился: {baseline.error_message} "
                "Он не использует признаки, поэтому его отказ указывает на общий путь данных, "
                "а не на модель. Остальные результаты в этом прогоне недостоверны."
            )
        elif self._cancel.is_set() and not self._record.error_code:
            self._record.error_code = "cancelled"
            self._record.error_message = "Прогон остановлен пользователем."

        self._record.finished_at = _utc_now()
        self._save()
        self._emit(
            ArenaEvent(
                kind="run_finished",
                at=now(),
                message=f"Прогон завершён: {self._record.state}.",
                payload={"state": self._record.state, "error_code": self._record.error_code},
            )
        )

        try:
            self._events.close()
        except (OSError, ValueError):
            return

    # ------------------------------------------------------------------ вспомогательное

    def _emit(self, event: ArenaEvent) -> None:
        self._log.append(event)

    def _save(self) -> None:
        self._store.save(self._record)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
