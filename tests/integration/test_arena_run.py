"""Прогон целиком, с настоящими процессами обучения.

Сценарии здесь — не «happy path и пара исключений», а именно те случаи, в которых движок
способен соврать: часть моделей упала, пользователь нажал «Остановить», контендер завис,
baseline не обучился. Проверять их без реальных процессов бессмысленно: отмена, аварийное
завершение и предел времени существуют ровно на границе процесса.
"""
from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from backend.arena.runner import ArenaRunner
from backend.arena.store import RunRecord
from backend.core.config import get_settings
from backend.datasets.registry import DatasetRegistry, DatasetSnapshot
from backend.models.context import ResourceBudget, build_context
from backend.models.registry import Contender, build_contenders
from backend.protocol.folds import FoldPlan, build_folds
from backend.protocol.recommend import ProtocolProposal, recommend_protocol
from backend.tasks.spec import TaskSpec, build_task_spec, propose_task_setup

BUDGET = ResourceBudget(max_parallel_fits=2, threads_per_fit=1, memory_budget_mb=1024)
#медленный участник нужен, чтобы отмена и предел времени успели сработать на живом обучении,
#а не на уже завершившемся
SLOW_FOREST = {"n_estimators": 4000, "max_depth": None, "min_samples_leaf": 1}
WAIT_SECONDS = 180


@dataclass
class Experiment:
    runner: ArenaRunner
    registry: DatasetRegistry
    snapshot: DatasetSnapshot
    frame: pl.DataFrame
    task: TaskSpec
    protocol: ProtocolProposal
    folds: FoldPlan
    contenders: list[Contender]

    def start(self, **overrides: object) -> tuple[RunRecord, bool]:
        arguments = {
            "dataset_id": self.snapshot.dataset_id,
            "dataset_fingerprint": self.snapshot.fingerprint,
            "dataset_path": self.registry.data_path(self.snapshot.dataset_id),
            "frame": self.frame,
            "task": self.task,
            "protocol": self.protocol,
            "folds": self.folds,
            "contenders": self.contenders,
            "budget": BUDGET,
        }
        arguments.update(overrides)
        return self.runner.start(**arguments)  # type: ignore[arg-type]

    def finish(self, run_id: str) -> RunRecord:
        self.runner.wait(run_id, timeout=WAIT_SECONDS)
        return self.runner.get(run_id)


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Experiment]:
    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    rows = 300
    source = tmp_path / "customers.parquet"
    pl.DataFrame(
        {
            "amount": [float((value * 41) % 233) for value in range(rows)],
            "tenure": [float(value % 37) for value in range(rows)],
            "plan": [["basic", "pro", "max"][value % 3] for value in range(rows)],
            "churned": [(value * 41) % 233 > 120 for value in range(rows)],
        }
    ).write_parquet(source)

    registry = DatasetRegistry()
    snapshot = registry.import_file(source, name="customers")
    frame = registry.load_frame(snapshot.dataset_id)
    proposal = propose_task_setup(snapshot, frame, "churned")
    task = build_task_spec(
        frame,
        target_column=proposal.target_column,
        task_type=proposal.task_type,
        feature_columns=proposal.feature_columns,
        positive_label=proposal.positive_label,
    )
    protocol = recommend_protocol(frame, task, n_splits=2, holdout_size=0.2, seed=7)
    folds = build_folds(frame, task, protocol)
    context = build_context(frame, task, n_train_rows=len(folds.train_pool))

    yield Experiment(
        runner=ArenaRunner(),
        registry=registry,
        snapshot=snapshot,
        frame=frame,
        task=task,
        protocol=protocol,
        folds=folds,
        contenders=build_contenders(context, selected_keys=["logistic_regression"]),
    )

    get_settings.cache_clear()


def _replace(contenders: list[Contender], key: str, **changes: object) -> list[Contender]:
    return [
        dataclasses.replace(item, **changes) if item.contender_key == key else item
        for item in contenders
    ]


def _await_running(experiment: Experiment, run_id: str, *, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        record = experiment.runner.get(run_id)

        if any(item.state == "RUNNING" for item in record.contenders):
            return

        if not experiment.runner.is_active(run_id):
            return

        time.sleep(0.1)

    raise AssertionError("Ни один контендер не дошёл до состояния RUNNING")


# ---------------------------------------------------------------- чистый прогон


def test_clean_run_succeeds_and_leaves_verifiable_predictions(experiment: Experiment) -> None:
    """Успешный прогон: состояние, метрики и предсказания, по которым метрики проверяемы."""
    record, duplicate = experiment.start()
    assert not duplicate

    final = experiment.finish(record.run_id)
    assert final.state == "SUCCEEDED"
    assert final.error_code is None
    assert {item.state for item in final.contenders} == {"SUCCEEDED"}

    for item in final.contenders:
        assert item.folds_completed == experiment.folds.n_splits
        assert item.coverage["complete"] is True
        assert item.metrics, f"{item.contender_key}: метрики не посчитаны"

        path = experiment.runner.store.predictions_path(record.run_id, item.contender_key)
        predictions = pl.read_parquet(path)
        pool = predictions.filter(pl.col("split") == "cv")

        #каждая строка обучающего пула получила ровно одно out-of-fold предсказание
        assert pool.height == len(experiment.folds.train_pool)
        assert pool["__row_id__"].n_unique() == pool.height
        #holdout считается один раз и отдельно
        assert predictions.filter(pl.col("split") == "holdout").height == len(
            experiment.folds.holdout
        )


def test_predictions_reference_snapshot_rows_not_filtered_positions(
    experiment: Experiment,
) -> None:
    #фолды строятся по строкам с известной целью, а предсказания подписываются строками
    #снимка: расхождение съело бы ровно число выброшенных строк — тихо и правдоподобно
    record, _ = experiment.start()
    final = experiment.finish(record.run_id)
    assert final.state == "SUCCEEDED"

    path = experiment.runner.store.predictions_path(record.run_id, "baseline_majority")
    predictions = pl.read_parquet(path)
    identifiers = predictions["__row_id__"].to_numpy()

    assert identifiers.min() >= 0
    assert identifiers.max() < experiment.snapshot.row_count


def test_every_contender_sees_the_same_folds(experiment: Experiment) -> None:
    """Одинаковые фолды у всех участников — техническая гарантия честности (D-7)."""
    record, _ = experiment.start()
    final = experiment.finish(record.run_id)

    assignments: list[set[tuple[int, int]]] = []

    for item in final.contenders:
        frame = pl.read_parquet(
            experiment.runner.store.predictions_path(record.run_id, item.contender_key)
        ).filter(pl.col("split") == "cv")
        assignments.append(set(zip(frame["__row_id__"].to_list(), frame["fold"].to_list(), strict=True)))

    assert len(assignments) > 1
    assert all(item == assignments[0] for item in assignments)


def test_no_training_process_outlives_the_run(experiment: Experiment) -> None:
    """После прогона не должно остаться ни одного живого дочернего процесса.

    Пул `joblib`, поднятый по `n_jobs`, однажды пережил своего создателя и держал
    прогон в RUNNING неопределённо долго. Проверка стоит здесь, чтобы это не вернулось.
    """
    import multiprocessing as mp

    before = {process.pid for process in mp.active_children()}
    record, _ = experiment.start()
    final = experiment.finish(record.run_id)

    assert final.state == "SUCCEEDED"
    leftover = {process.pid for process in mp.active_children()} - before
    assert not leftover, f"после прогона остались процессы: {leftover}"


# ---------------------------------------------------------------- частичный успех


def test_failed_contender_does_not_destroy_finished_results(experiment: Experiment) -> None:
    """Падение одной модели — отказ участника, а не прогона."""
    #недопустимое число итераций: sklearn отвергнет его при обучении, а не при сборке
    contenders = _replace(experiment.contenders, "logistic_regression", params={"max_iter": -1})
    record, _ = experiment.start(contenders=contenders)
    final = experiment.finish(record.run_id)

    assert final.state == "PARTIAL"
    broken = final.contender("logistic_regression")
    assert broken.state == "FAILED"
    assert broken.error_code == "training_failed"
    #наружу уходит код и одна строка, без путей и внутренностей окружения
    assert "Traceback" not in broken.error_message
    assert ":\\" not in broken.error_message

    baseline = final.contender("baseline_majority")
    assert baseline.state == "SUCCEEDED"
    assert baseline.metrics


def test_failed_baseline_invalidates_the_whole_run(experiment: Experiment) -> None:
    """Baseline не смотрит на признаки: его отказ указывает на путь данных."""
    contenders = _replace(
        experiment.contenders, "baseline_majority", adapter_key="adapter_that_does_not_exist"
    )
    record, _ = experiment.start(contenders=contenders)
    final = experiment.finish(record.run_id)

    assert final.state == "FAILED"
    assert final.error_code == "baseline_failed"
    assert "путь данных" in final.error_message
    #результат остальных при этом сохранён: он есть, но прогон объявлен недостоверным
    assert final.contender("logistic_regression").state == "SUCCEEDED"


# ---------------------------------------------------------------- остановка и пределы


def test_cancel_stops_training_and_keeps_finished_results(experiment: Experiment) -> None:
    """Остановка настоящая: процессы завершаются, а посчитанное сохраняется."""
    contenders = _replace(
        experiment.contenders,
        "logistic_regression",
        adapter_key="random_forest",
        preprocessing_profile="tree_ordinal",
        params=SLOW_FOREST,
    )
    record, _ = experiment.start(contenders=contenders)
    _await_running(experiment, record.run_id)

    experiment.runner.cancel(record.run_id)
    final = experiment.finish(record.run_id)

    assert final.state == "CANCELLED"
    assert final.error_code == "cancelled"
    assert not experiment.runner.is_active(record.run_id)
    #ни один участник не остался в незавершённом состоянии
    assert all(
        item.state in {"SUCCEEDED", "FAILED", "SKIPPED", "CANCELLED"} for item in final.contenders
    )

    stopped = final.contender("logistic_regression")
    assert stopped.state == "CANCELLED"
    assert stopped.error_code == "cancelled"


def test_cancelling_a_finished_run_is_refused(experiment: Experiment) -> None:
    from backend.core.errors import ValidationError

    record, _ = experiment.start()
    final = experiment.finish(record.run_id)
    assert final.state == "SUCCEEDED"

    with pytest.raises(ValidationError, match="завершён"):
        experiment.runner.cancel(record.run_id)


def test_contender_over_its_time_limit_is_stopped_with_a_reason(
    experiment: Experiment,
) -> None:
    contenders = _replace(
        experiment.contenders,
        "logistic_regression",
        adapter_key="random_forest",
        preprocessing_profile="tree_ordinal",
        params=SLOW_FOREST,
    )
    record, _ = experiment.start(contenders=contenders, contender_timeout_seconds=3)
    final = experiment.finish(record.run_id)

    slow = final.contender("logistic_regression")
    assert slow.state == "FAILED"
    assert slow.error_code == "timeout"
    #предел времени у одного участника не отменяет результат остальных
    assert final.contender("baseline_majority").state == "SUCCEEDED"
    assert final.state == "PARTIAL"


# ---------------------------------------------------------------- повторный запуск


def test_repeated_submission_returns_the_running_run(experiment: Experiment) -> None:
    """Двойная отправка не создаёт второй такой же прогон."""
    contenders = _replace(
        experiment.contenders,
        "logistic_regression",
        adapter_key="random_forest",
        preprocessing_profile="tree_ordinal",
        params=SLOW_FOREST,
    )
    first, duplicate_first = experiment.start(contenders=contenders)
    _await_running(experiment, first.run_id)
    second, duplicate_second = experiment.start(contenders=contenders)

    assert not duplicate_first
    assert duplicate_second
    assert second.run_id == first.run_id
    assert len(experiment.runner.list_runs()) == 1

    experiment.runner.cancel(first.run_id)
    experiment.finish(first.run_id)


def test_history_and_results_survive_a_new_runner(experiment: Experiment) -> None:
    """Перезапуск backend не теряет ни истории, ни предсказаний."""
    record, _ = experiment.start()
    final = experiment.finish(record.run_id)
    assert final.state == "SUCCEEDED"

    #новый экземпляр — это то же, что новый процесс backend поверх тех же артефактов
    restarted = ArenaRunner()
    restored = restarted.get(record.run_id)

    assert restored.state == "SUCCEEDED"
    assert [item.state for item in restored.contenders] == [
        item.state for item in final.contenders
    ]
    assert restored.contenders[0].metrics
    assert restarted.events(record.run_id), "журнал событий не читается после перезапуска"
    assert pl.read_parquet(
        restarted.store.predictions_path(record.run_id, "baseline_majority")
    ).height > 0


def test_progress_counts_only_completed_folds(experiment: Experiment) -> None:
    record, _ = experiment.start()
    final = experiment.finish(record.run_id)
    progress = experiment.runner.progress(record.run_id).to_dict()

    assert final.state == "SUCCEEDED"
    assert progress["folds_completed"] == progress["folds_planned"]
    assert progress["completed_fraction"] == 1.0
    kinds = [event.kind for event in experiment.runner.events(record.run_id)]
    assert kinds[0] == "run_queued"
    assert kinds[-1] == "run_finished"
    assert kinds.count("fold_completed") == progress["folds_planned"]


def test_holdout_rows_are_never_used_for_training(experiment: Experiment) -> None:
    """Ни один фолд не обучается на строках holdout."""
    holdout = set(experiment.folds.holdout.tolist())

    for training, validation in experiment.folds.folds:
        assert not holdout & set(training.tolist())
        assert not holdout & set(validation.tolist())

    #и обучение не пересекается с проверкой — это движок проверяет и сам, до обучения
    for training, validation in experiment.folds.folds:
        assert not np.intersect1d(training, validation).size


def test_fitting_on_validation_rows_would_be_visible(tmp_path: Path, monkeypatch) -> None:
    """Обучение внутри фолда не должно видеть проверочные строки.

    Проверяется поведением, а не структурой. На данных без всякой связи между
    признаками и целью ближайший сосед при `k=1` отвечает случайно — если он обучен
    только на обучающей части. Если же обучающая часть включает проверочные строки,
    сосед находит **саму проверяемую строку** и попадает в цель всегда. Разрыв между
    случайным угадыванием и единицей ни с чем не спутать.
    """
    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    rows = 240
    generator = np.random.default_rng(17)
    source = tmp_path / "noise.parquet"
    #связи между признаками и целью нет: любая честная модель здесь около случайной
    pl.DataFrame(
        {
            "first": generator.normal(size=rows),
            "second": generator.normal(size=rows),
            "third": generator.normal(size=rows),
            "label": generator.integers(0, 2, size=rows).astype(bool),
        }
    ).write_parquet(source)

    registry = DatasetRegistry()
    snapshot = registry.import_file(source, name="noise")
    frame = registry.load_frame(snapshot.dataset_id)
    proposal = propose_task_setup(snapshot, frame, "label")
    task = build_task_spec(
        frame,
        target_column="label",
        task_type=proposal.task_type,
        feature_columns=proposal.feature_columns,
        positive_label=proposal.positive_label,
    )
    protocol = recommend_protocol(frame, task, n_splits=2, holdout_size=0.2, seed=3)
    folds = build_folds(frame, task, protocol)
    context = build_context(frame, task, n_train_rows=len(folds.train_pool))
    contenders = _replace(
        build_contenders(context, selected_keys=["knn"]), "knn", params={"n_neighbors": 1}
    )

    runner = ArenaRunner()
    record, _ = runner.start(
        dataset_id=snapshot.dataset_id,
        dataset_fingerprint=snapshot.fingerprint,
        dataset_path=registry.data_path(snapshot.dataset_id),
        frame=frame,
        task=task,
        protocol=protocol,
        folds=folds,
        contenders=contenders,
        budget=BUDGET,
    )
    runner.wait(record.run_id, timeout=WAIT_SECONDS)
    final = runner.get(record.run_id)
    assert final.state == "SUCCEEDED", final.error_message

    neighbour = final.contender("knn")
    accuracy = next(
        metric
        for part in neighbour.metrics
        if part["split"] == "cv"
        for metric in part["metrics"]
        if metric["key"] == "accuracy"
    )
    assert accuracy["value"] is not None
    assert accuracy["value"] < 0.85, (
        "Ближайший сосед угадывает шум — значит обучение видело проверочные строки"
    )

    get_settings.cache_clear()
