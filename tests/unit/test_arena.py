"""Семантика прогона без обучения: состояния, вероятности, покрытие, метрики, хранилище.

Тесты здесь проверяют утверждения, ошибка в которых даёт **правдоподобное** число.
Съехавший на один столбец порядок классов, удвоенное out-of-fold предсказание и метрика,
усреднённая по неполному набору фолдов, выглядят нормально ровно до того момента, когда
на их основе выбирают модель.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from backend.arena.events import ArenaEvent, EventLog, Progress
from backend.arena.metrics import evaluate, primary_metric
from backend.arena.predictions import (
    HOLDOUT_FOLD,
    SPLIT_CV,
    SPLIT_HOLDOUT,
    FoldPredictions,
    PredictionError,
    align_probabilities,
    to_frame,
    validate_fold_predictions,
    verify_oof_coverage,
)
from backend.arena.spec import build_run_spec
from backend.arena.states import resolve_run_state
from backend.arena.store import ContenderRecord, RunRecord, RunStore, new_run_id
from backend.core.errors import ValidationError
from backend.models.context import ResourceBudget, build_context
from backend.models.registry import build_contenders
from backend.protocol.folds import build_folds
from backend.protocol.recommend import recommend_protocol
from backend.tasks.spec import build_task_spec

LABELS = ["no", "yes"]


# ---------------------------------------------------------------- состояния прогона


def test_partial_is_not_success() -> None:
    """Прогон с упавшим участником не имеет права выглядеть удачным."""
    state = resolve_run_state(
        contender_states=["SUCCEEDED", "SUCCEEDED", "FAILED"],
        cancelled=False,
        baseline_failed=False,
    )

    assert state == "PARTIAL"


def test_skipped_is_not_counted_as_failure() -> None:
    #пропущенный по применимости участник исключён до старта и с причиной:
    #в знаменатель неудач он не входит
    state = resolve_run_state(
        contender_states=["SUCCEEDED", "SKIPPED", "SKIPPED"],
        cancelled=False,
        baseline_failed=False,
    )

    assert state == "SUCCEEDED"


def test_cancel_names_the_real_reason_even_with_results() -> None:
    #прогон неполон потому, что его остановили, а не потому, что модели проиграли
    state = resolve_run_state(
        contender_states=["SUCCEEDED", "CANCELLED"], cancelled=True, baseline_failed=False
    )

    assert state == "CANCELLED"


def test_failed_baseline_overrides_every_success() -> None:
    """Baseline не смотрит на признаки: его отказ — это отказ пути данных."""
    state = resolve_run_state(
        contender_states=["SUCCEEDED", "SUCCEEDED", "FAILED"],
        cancelled=False,
        baseline_failed=True,
    )

    assert state == "FAILED"


def test_run_without_a_single_result_is_failed() -> None:
    state = resolve_run_state(
        contender_states=["FAILED", "SKIPPED"], cancelled=False, baseline_failed=False
    )

    assert state == "FAILED"


# ---------------------------------------------------------------- вероятности


def test_probabilities_follow_the_canonical_label_order() -> None:
    """Столбцы переставляются по меткам, а не берутся в порядке модели."""
    #модель отдала классы в обратном порядке: без выравнивания «yes» читался бы как «no»
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    aligned, missing = align_probabilities(proba, ["yes", "no"], LABELS)

    assert missing == []
    assert aligned[0].tolist() == [0.1, 0.9]
    assert aligned[1].tolist() == [0.8, 0.2]


def test_class_absent_from_training_gets_a_zero_column_and_is_reported() -> None:
    proba = np.array([[1.0], [1.0]])
    aligned, missing = align_probabilities(proba, ["no"], LABELS)

    assert missing == ["yes"]
    assert aligned[:, 1].tolist() == [0.0, 0.0]


def test_class_outside_the_task_is_refused_rather_than_dropped() -> None:
    #выброшенный столбец дал бы вероятности, не сходящиеся к единице, без объяснения
    with pytest.raises(PredictionError, match="вне постановки"):
        align_probabilities(np.array([[0.3, 0.3, 0.4]]), ["no", "yes", "maybe"], LABELS)


# ---------------------------------------------------------------- форма предсказаний


def _classification_fold(*, rows: int = 4, fold: int = 0) -> FoldPredictions:
    return FoldPredictions(
        row_ids=np.arange(rows),
        fold=fold,
        split=SPLIT_CV,
        y_true=np.array(["no", "yes"] * (rows // 2), dtype=object),
        y_pred=np.array(["no", "yes"] * (rows // 2), dtype=object),
        proba=np.tile(np.array([[0.8, 0.2], [0.3, 0.7]]), (rows // 2, 1)),
    )


def test_wrong_prediction_count_is_refused() -> None:
    with pytest.raises(PredictionError, match="предсказаний на"):
        validate_fold_predictions(
            _classification_fold(), expected_rows=5, task_type="binary", class_labels=LABELS
        )


def test_non_finite_regression_prediction_is_refused() -> None:
    part = FoldPredictions(
        row_ids=np.arange(3),
        fold=0,
        split=SPLIT_CV,
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.0, np.nan, np.inf]),
        proba=None,
    )

    with pytest.raises(PredictionError, match="NaN"):
        validate_fold_predictions(
            part, expected_rows=3, task_type="regression", class_labels=[]
        )


def test_label_outside_the_task_is_refused() -> None:
    part = _classification_fold()
    part = FoldPredictions(
        row_ids=part.row_ids,
        fold=0,
        split=SPLIT_CV,
        y_true=part.y_true,
        y_pred=np.array(["no", "maybe", "no", "yes"], dtype=object),
        proba=part.proba,
    )

    with pytest.raises(PredictionError, match="вне постановки"):
        validate_fold_predictions(
            part, expected_rows=4, task_type="binary", class_labels=LABELS
        )


# ---------------------------------------------------------------- покрытие out-of-fold


def _coverage_frame(row_ids: list[int], folds: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "__row_id__": np.array(row_ids, dtype=np.int64),
            "split": [SPLIT_CV] * len(row_ids),
            "fold": np.array(folds, dtype=np.int32),
            "y_true": ["no"] * len(row_ids),
            "y_pred": ["no"] * len(row_ids),
        }
    )


def test_complete_coverage_is_accepted() -> None:
    report = verify_oof_coverage(
        _coverage_frame([0, 1, 2, 3], [0, 0, 1, 1]),
        expected_row_ids=np.array([0, 1, 2, 3]),
        strategy="k_fold",
    )

    assert report.complete
    assert report.duplicated_rows == 0


def test_duplicated_out_of_fold_prediction_is_refused() -> None:
    """Строка, попавшая в проверку дважды, удваивает свой вклад в метрику."""
    with pytest.raises(PredictionError, match="больше одного"):
        verify_oof_coverage(
            _coverage_frame([0, 1, 1, 2], [0, 0, 1, 1]),
            expected_row_ids=np.array([0, 1, 2]),
            strategy="k_fold",
        )


def test_incomplete_coverage_is_refused_for_k_fold() -> None:
    with pytest.raises(PredictionError, match="из"):
        verify_oof_coverage(
            _coverage_frame([0, 1], [0, 0]),
            expected_row_ids=np.array([0, 1, 2, 3]),
            strategy="stratified_k_fold",
        )


def test_incomplete_coverage_is_correct_for_rolling_window() -> None:
    """При скользящем окне самый ранний отрезок только обучает и проверочным не бывает."""
    report = verify_oof_coverage(
        _coverage_frame([2, 3], [0, 1]),
        expected_row_ids=np.array([0, 1, 2, 3]),
        strategy="forward_chaining",
    )

    assert not report.complete
    assert "скользящем окне" in report.note


def test_prediction_for_a_row_outside_the_pool_is_refused() -> None:
    #строка из holdout, попавшая в cv, означает, что модель видела проверочные данные
    with pytest.raises(PredictionError, match="вне обучающего пула"):
        verify_oof_coverage(
            _coverage_frame([0, 1, 99], [0, 0, 1]),
            expected_row_ids=np.array([0, 1]),
            strategy="k_fold",
        )


# ---------------------------------------------------------------- метрики


def _prediction_frame() -> pl.DataFrame:
    rows = 40
    truth = ["yes" if index % 2 else "no" for index in range(rows)]
    scores = [0.9 if value == "yes" else 0.1 for value in truth]

    return pl.DataFrame(
        {
            "__row_id__": np.arange(rows, dtype=np.int64),
            "split": [SPLIT_CV] * rows,
            #фолд задаётся половиной таблицы, а не чётностью строки: при чётности
            #в каждом фолде оказался бы ровно один класс, и ранговые метрики стали бы
            #неопределимы — фикстура проверяла бы не то, что заявлено
            "fold": np.array([0] * (rows // 2) + [1] * (rows // 2), dtype=np.int32),
            "y_true": truth,
            "y_pred": truth,
            "proba__no": np.array([1.0 - value for value in scores], dtype=np.float32),
            "proba__yes": np.array(scores, dtype=np.float32),
        }
    )


def test_metrics_come_from_predictions_only() -> None:
    sets = evaluate(
        _prediction_frame(),
        task_type="binary",
        class_labels=LABELS,
        positive_label="yes",
    )
    cross_validated = sets[0]

    assert cross_validated.split == SPLIT_CV
    assert cross_validated.by_key("roc_auc").value == 1.0
    assert cross_validated.by_key("accuracy").value == 1.0
    #среднее по фолдам, а не по объединённым строкам: разброс должен быть виден
    assert cross_validated.by_key("roc_auc").per_fold == [1.0, 1.0]


def test_undefined_metric_is_none_with_a_reason_not_zero() -> None:
    """ROC-AUC на части с одним классом не равен нулю — он не определён."""
    rows = 6
    frame = pl.DataFrame(
        {
            "__row_id__": np.arange(rows, dtype=np.int64),
            "split": [SPLIT_CV] * rows,
            "fold": np.zeros(rows, dtype=np.int32),
            "y_true": ["no"] * rows,
            "y_pred": ["no"] * rows,
            "proba__no": np.full(rows, 0.9, dtype=np.float32),
            "proba__yes": np.full(rows, 0.1, dtype=np.float32),
        }
    )
    sets = evaluate(frame, task_type="binary", class_labels=LABELS, positive_label="yes")
    roc = sets[0].by_key("roc_auc")

    assert roc.value is None
    assert "один класс" in roc.note


def test_probabilities_that_do_not_sum_to_one_are_refused_with_the_real_reason() -> None:
    #съехавшее выравнивание столбцов даёт правдоподобное число: единственный признак
    #беды — потерянная масса вероятности, и молчать о ней нельзя
    rows = 4
    frame = pl.DataFrame(
        {
            "__row_id__": np.arange(rows, dtype=np.int64),
            "split": [SPLIT_CV] * rows,
            "fold": np.zeros(rows, dtype=np.int32),
            "y_true": ["no", "yes", "no", "yes"],
            "y_pred": ["no", "yes", "no", "yes"],
            "proba__no": np.full(rows, 0.3, dtype=np.float32),
            "proba__yes": np.full(rows, 0.3, dtype=np.float32),
        }
    )
    sets = evaluate(frame, task_type="binary", class_labels=LABELS, positive_label="yes")
    log_loss = sets[0].by_key("log_loss")

    assert log_loss.value is None
    assert "не суммируются" in log_loss.note


def test_holdout_metrics_are_separate_from_cross_validation() -> None:
    cv = _prediction_frame()
    holdout = cv.head(8).with_columns(
        pl.lit(SPLIT_HOLDOUT).alias("split"),
        pl.lit(HOLDOUT_FOLD, dtype=pl.Int32).alias("fold"),
    )
    sets = evaluate(
        pl.concat([cv, holdout]),
        task_type="binary",
        class_labels=LABELS,
        positive_label="yes",
    )

    assert [item.split for item in sets] == [SPLIT_CV, SPLIT_HOLDOUT]


def test_regression_metrics_refuse_mape_on_zero_target() -> None:
    rows = 6
    frame = pl.DataFrame(
        {
            "__row_id__": np.arange(rows, dtype=np.int64),
            "split": [SPLIT_CV] * rows,
            "fold": np.zeros(rows, dtype=np.int32),
            "y_true": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            "y_pred": np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]),
        }
    )
    sets = evaluate(frame, task_type="regression", class_labels=[], positive_label=None)

    assert sets[0].by_key("mape").value is None
    assert "нулевой целью" in sets[0].by_key("mape").note
    assert sets[0].by_key("rmse").value is not None


def test_primary_metric_is_declared_per_task() -> None:
    assert primary_metric("binary") == "roc_auc"
    assert primary_metric("regression") == "rmse"


def test_prediction_table_keeps_snapshot_row_identity() -> None:
    part = _classification_fold()
    frame = to_frame([part], task_type="binary", class_labels=LABELS)

    assert frame["__row_id__"].to_list() == [0, 1, 2, 3]
    assert set(frame.columns) >= {"__row_id__", "split", "fold", "y_true", "y_pred"}
    assert "proba__yes" in frame.columns


# ---------------------------------------------------------------- отпечаток эксперимента


def _experiment(seed: int = 42, n_splits: int = 3):
    rows = 200
    frame = pl.DataFrame(
        {
            "amount": [float(value % 50) for value in range(rows)],
            "plan": [["basic", "pro"][value % 2] for value in range(rows)],
            "churned": [value % 3 == 0 for value in range(rows)],
        }
    )
    task = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["amount", "plan"],
        positive_label="True",
    )
    protocol = recommend_protocol(frame, task, n_splits=n_splits, holdout_size=0.2, seed=seed)
    folds = build_folds(frame, task, protocol)
    context = build_context(frame, task, n_train_rows=len(folds.train_pool))
    contenders = build_contenders(context, selected_keys=["logistic_regression"])
    return frame, task, protocol, folds, contenders


def _spec(budget: ResourceBudget, seed: int = 42):
    _, task, protocol, folds, contenders = _experiment(seed=seed)
    return build_run_spec(
        dataset_id="ds_0123456789abcdef",
        dataset_fingerprint="fp",
        task=task,
        protocol=protocol,
        folds=folds,
        contenders=contenders,
        budget=budget,
    )


def test_same_experiment_gives_the_same_fingerprint() -> None:
    budget = ResourceBudget(max_parallel_fits=2, threads_per_fit=2, memory_budget_mb=1024)

    assert _spec(budget).experiment_fingerprint == _spec(budget).experiment_fingerprint


def test_resources_are_not_part_of_experiment_identity() -> None:
    """Тот же эксперимент на другой машине — тот же эксперимент."""
    small = ResourceBudget(max_parallel_fits=1, threads_per_fit=1, memory_budget_mb=512)
    large = ResourceBudget(max_parallel_fits=8, threads_per_fit=8, memory_budget_mb=65536)

    assert _spec(small).experiment_fingerprint == _spec(large).experiment_fingerprint


def test_different_seed_is_a_different_experiment() -> None:
    budget = ResourceBudget(max_parallel_fits=1, threads_per_fit=1, memory_budget_mb=512)

    assert _spec(budget, seed=1).experiment_fingerprint != _spec(budget, seed=2).experiment_fingerprint


def test_spec_carries_the_fold_assignment_hash() -> None:
    #хеш разбиения переносится в карточку прогона: честность фолдов должна быть проверяемой,
    #а не обещанной
    _, _, _, folds, _ = _experiment()
    budget = ResourceBudget(max_parallel_fits=1, threads_per_fit=1, memory_budget_mb=512)
    spec = _spec(budget)

    assert spec.fold_assignment_hash == folds.assignment_hash
    assert spec.baseline_key == "baseline_majority"


# ---------------------------------------------------------------- хранилище


def _record(store: RunStore, run_id: str) -> RunRecord:
    budget = ResourceBudget(max_parallel_fits=1, threads_per_fit=1, memory_budget_mb=512)
    spec = _spec(budget)
    return store.create(
        run_id,
        spec,
        [
            ContenderRecord(
                contender_key=contender.contender_key,
                label=contender.label,
                family=contender.family,
                preprocessing_profile=contender.preprocessing_profile,
                is_baseline=contender.is_baseline,
            )
            for contender in spec.contenders
        ],
    )


def test_result_becomes_visible_only_as_a_whole(tmp_path: Path) -> None:
    """Публикация — один перенос каталога: метрики без предсказаний невозможны."""
    store = RunStore(root=tmp_path)
    run_id = new_run_id()
    _record(store, run_id)

    staging = store.staging_for(run_id, "logistic_regression")
    (staging / "result.json").write_text('{"status": "SUCCEEDED"}', encoding="utf-8")
    (staging / "predictions.parquet").write_bytes(b"parquet")

    final = store.directory(run_id) / "contenders" / "logistic_regression"
    assert not final.exists()

    store.publish(run_id, "logistic_regression")
    assert (final / "result.json").exists()
    assert (final / "predictions.parquet").exists()
    assert not staging.exists()


def test_contender_key_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    run_id = new_run_id()
    _record(store, run_id)

    for hostile in ("../../etc", "a/b", "logistic..regression"):
        with pytest.raises(ValidationError):
            store.staging_for(run_id, hostile)


def test_run_id_is_validated_before_touching_the_filesystem(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)

    with pytest.raises(ValidationError):
        store.directory("../secrets")


def test_restart_turns_a_stuck_running_into_a_terminal_state(tmp_path: Path) -> None:
    """RUNNING, переживший перезапуск, обязан получить честный конец, а не висеть вечно."""
    store = RunStore(root=tmp_path)
    run_id = new_run_id()
    record = _record(store, run_id)
    record.state = "RUNNING"
    #метка чужого процесса: так выглядит запись, оставшаяся от прошлого backend
    record.owner_token = "another-process"  # noqa: S105
    record.contenders[0].state = "SUCCEEDED"
    record.contenders[1].state = "RUNNING"
    store.save(record)

    assert store.recover_interrupted() == [run_id]

    recovered = store.get(run_id)
    assert recovered.state == "INTERRUPTED"
    #уже посчитанный результат сохраняется: он получен честно, и терять его незачем
    assert recovered.contenders[0].state == "SUCCEEDED"
    assert recovered.contenders[1].state == "CANCELLED"
    assert recovered.contenders[1].error_code == "interrupted"


def test_own_running_run_is_not_touched_by_recovery(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    run_id = new_run_id()
    record = _record(store, run_id)
    record.state = "RUNNING"
    store.save(record)

    assert store.recover_interrupted() == []
    assert store.get(run_id).state == "RUNNING"


def test_broken_run_card_does_not_hide_the_rest(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    good = new_run_id()
    _record(store, good)
    broken = store.root / new_run_id()
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{ не json", encoding="utf-8")

    assert [item.run_id for item in store.list_runs()] == [good]


def test_active_run_is_found_by_experiment_fingerprint(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    run_id = new_run_id()
    record = _record(store, run_id)
    fingerprint = record.experiment_fingerprint

    assert store.find_active_by_fingerprint(fingerprint).run_id == run_id

    record.state = "SUCCEEDED"
    store.save(record)
    #завершённый прогон повторить можно: это осознанное воспроизведение, а не случайность
    assert store.find_active_by_fingerprint(fingerprint) is None


# ---------------------------------------------------------------- прогресс и журнал


def test_progress_has_no_fraction_when_nothing_is_planned() -> None:
    #ноль в знаменателе не заменяется на «0%»: это разные утверждения
    assert Progress(0, 0, 0, 0).to_dict()["completed_fraction"] is None
    assert Progress(2, 8, 1, 3).to_dict()["completed_fraction"] == 0.25


def test_event_log_survives_a_truncated_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path=path)
    log.append(ArenaEvent(kind="run_started", at="2026-01-01T00:00:00Z", message="начали"))
    #обрыв записи портит последнюю строку, а не весь журнал
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"kind": "fold_comp')

    restored = EventLog.load(path)
    assert [event.kind for event in restored.snapshot()] == ["run_started"]


def test_event_log_reads_back_what_it_wrote(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path=path)
    log.append(ArenaEvent(kind="run_started", at="t", message="начали"))
    log.append(ArenaEvent(kind="fold_completed", at="t", message="фолд", fold=1, n_folds=3))

    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["kind"] for item in payloads] == ["run_started", "fold_completed"]
    assert log.snapshot(since=1)[0].fold == 1
