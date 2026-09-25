"""Честный выбор модели: что именно значит «эта лучше той».

Проверяются не формулы, а утверждения, из-за которых таблица результатов способна
обмануть: победитель, чьё превосходство не держится на фолдах; holdout, попавший
в ранжирование; участник без метрики, оказавшийся последним с нулём; порядок,
меняющийся от того, кого ещё запустили.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from backend.arena.predictions import SPLIT_CV, SPLIT_HOLDOUT
from backend.arena.store import ContenderRecord, RunRecord, RunStore, new_run_id
from backend.core.errors import ValidationError
from backend.scoring.compare import compare_per_fold
from backend.scoring.leaderboard import build_leaderboard
from backend.scoring.objective import build_objective, default_objective

LABELS = ["no", "yes"]
ROWS_PER_FOLD = 40


# ---------------------------------------------------------------- парное сравнение


def test_shared_fold_difficulty_cancels_out() -> None:
    """Трудный фолд роняет всех участников сразу и не должен читаться как неустойчивость.

    У обеих моделей разброс между фолдами большой, но разница между ними постоянна.
    Сравнение средних с их разбросами объявило бы обе неразличимыми; пофолдное
    сравнение видит устойчивое превосходство.
    """
    hard_folds = [0.90, 0.60, 0.75, 0.55, 0.85]
    better = [value + 0.05 for value in hard_folds]

    comparison = compare_per_fold(
        leader_key="a",
        leader_folds=better,
        trailing_key="b",
        trailing_folds=hard_folds,
        metric="ROC-AUC",
        higher_is_better=True,
    )

    assert comparison is not None
    assert comparison.stable
    assert comparison.folds_won == 5
    assert comparison.mean_difference == pytest.approx(0.05)
    #разброс самой разницы нулевой, хотя разброс каждой модели по фолдам огромен
    assert comparison.std_difference == pytest.approx(0.0, abs=1e-9)


def test_win_on_average_with_a_flipped_fold_is_not_stable() -> None:
    #выигрыш на четырёх фолдах и крупный проигрыш на пятом даёт положительное среднее,
    #но превосходство на разбиении не держится
    comparison = compare_per_fold(
        leader_key="a",
        #разницы +0.40 на четырёх фолдах и -0.10 на пятом: среднее крупное и превышает
        #разброс, но знак на разбиении не выдержан
        leader_folds=[0.90, 0.90, 0.90, 0.90, 0.50],
        trailing_key="b",
        trailing_folds=[0.50, 0.50, 0.50, 0.50, 0.60],
        metric="ROC-AUC",
        higher_is_better=True,
    )

    assert comparison is not None
    assert comparison.mean_difference > 0
    assert not comparison.stable
    assert "знак меняется" in comparison.explanation


def test_lower_is_better_metric_flips_the_sign() -> None:
    comparison = compare_per_fold(
        leader_key="a",
        leader_folds=[0.10, 0.12, 0.11],
        trailing_key="b",
        trailing_folds=[0.30, 0.32, 0.31],
        metric="RMSE",
        higher_is_better=False,
    )

    assert comparison is not None
    #меньшая ошибка — это превосходство, а не отставание
    assert comparison.mean_difference > 0
    assert comparison.stable


def test_nothing_to_compare_is_not_a_draw() -> None:
    assert (
        compare_per_fold(
            leader_key="a",
            leader_folds=[None, None],
            trailing_key="b",
            trailing_folds=[0.5, 0.6],
            metric="ROC-AUC",
            higher_is_better=True,
        )
        is None
    )


# ---------------------------------------------------------------- цель


def test_objective_refuses_a_metric_the_task_does_not_have() -> None:
    with pytest.raises(ValidationError, match="не считается для задачи"):
        build_objective("regression", metric="roc_auc")


def test_objective_describes_itself_in_words() -> None:
    objective = build_objective(
        "binary",
        metric="recall",
        constraints=[
            {"metric": "precision", "operator": "gte", "value": 0.8, "reason": "ложная тревога дорога"}
        ],
        min_gain_over_baseline=0.02,
    )
    described = objective.describe()

    assert "recall" in described
    assert "precision не ниже 0.8" in described
    assert "ложная тревога дорога" in described


# ---------------------------------------------------------------- таблица


def _predictions(
    fold_scores: list[float],
    *,
    holdout_score: float | None = None,
    with_proba: bool = True,
) -> pl.DataFrame:
    """Предсказания, дающие заданную долю правильных ответов на каждом фолде."""
    rows: list[dict] = []

    for fold, accuracy in enumerate(fold_scores):
        correct = round(accuracy * ROWS_PER_FOLD)

        for index in range(ROWS_PER_FOLD):
            truth = LABELS[index % 2]
            predicted = truth if index < correct else LABELS[(index + 1) % 2]
            confidence = 0.9 if predicted == "yes" else 0.1
            rows.append(
                {
                    "__row_id__": fold * ROWS_PER_FOLD + index,
                    "split": SPLIT_CV,
                    "fold": fold,
                    "y_true": truth,
                    "y_pred": predicted,
                    "proba__no": 1.0 - confidence,
                    "proba__yes": confidence,
                }
            )

    if holdout_score is not None:
        correct = round(holdout_score * ROWS_PER_FOLD)

        for index in range(ROWS_PER_FOLD):
            truth = LABELS[index % 2]
            predicted = truth if index < correct else LABELS[(index + 1) % 2]
            confidence = 0.9 if predicted == "yes" else 0.1
            rows.append(
                {
                    "__row_id__": 10_000 + index,
                    "split": SPLIT_HOLDOUT,
                    "fold": -1,
                    "y_true": truth,
                    "y_pred": predicted,
                    "proba__no": 1.0 - confidence,
                    "proba__yes": confidence,
                }
            )

    frame = pl.DataFrame(rows)
    frame = frame.with_columns(pl.col("fold").cast(pl.Int32))

    if not with_proba:
        #модель без вероятностей — обычное дело: SVM без probability, дерево решений.
        #Ранговые метрики для неё не определены, и это не «плохой результат»
        return frame.drop("proba__no", "proba__yes")

    return frame.with_columns(
        pl.col("proba__no").cast(pl.Float32),
        pl.col("proba__yes").cast(pl.Float32),
    )


def _run_with(
    store: RunStore, contenders: dict[str, dict], *, task_type: str = "binary"
) -> RunRecord:
    """Собрать прогон с готовыми предсказаниями, ничего не обучая."""
    run_id = new_run_id()
    records = [
        ContenderRecord(
            contender_key=key,
            label=spec.get("label", key),
            family=spec.get("family", "linear"),
            preprocessing_profile="scaled_onehot",
            is_baseline=spec.get("is_baseline", False),
            state=spec.get("state", "SUCCEEDED"),
            error_message=spec.get("error_message", ""),
            elapsed_seconds=spec.get("elapsed_seconds"),
            artifact=(
                {"model_bytes": spec["model_bytes"]}
                if spec.get("model_bytes") is not None
                else None
            ),
        )
        for key, spec in contenders.items()
    ]
    directory = store.directory(run_id)
    (directory / "contenders").mkdir(parents=True, exist_ok=True)
    (directory / ".staging").mkdir(parents=True, exist_ok=True)

    record = RunRecord(
        run_id=run_id,
        created_at="2026-09-13T00:00:00+00:00",
        state="SUCCEEDED",
        spec={
            "task": {
                "task_type": task_type,
                "class_labels": LABELS,
                "positive_label": "yes",
                "target_column": "target",
            },
            "contenders": [
                {"contender_key": key, "estimated_cost": spec.get("cost", 1.0)}
                for key, spec in contenders.items()
            ],
        },
        contenders=records,
    )
    store.save(record)

    for key, spec in contenders.items():
        if spec.get("state", "SUCCEEDED") != "SUCCEEDED":
            continue

        target = directory / "contenders" / key
        target.mkdir(parents=True, exist_ok=True)
        _predictions(
            spec["folds"],
            holdout_score=spec.get("holdout"),
            with_proba=spec.get("with_proba", True),
        ).write_parquet(target / "predictions.parquet")

    return record


def test_leaderboard_keeps_measured_cost_separate_from_estimate(tmp_path: Path) -> None:
    """Факт после обучения не подменяется оценкой, известной до запуска."""
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {
                "folds": [0.50, 0.50, 0.50],
                "is_baseline": True,
            },
            "model": {
                "folds": [0.90, 0.90, 0.90],
                "cost": 99.0,
                "elapsed_seconds": 8.4,
                "model_bytes": 44_040_192,
            },
        },
    )

    board = build_leaderboard(
        record=record, store=store, objective=default_objective("binary")
    )
    row = next(item for item in board.rows if item.contender_key == "model")

    assert row.estimated_cost == 99.0
    assert row.elapsed_seconds == 8.4
    assert row.model_bytes == 44_040_192


def test_champion_requires_superiority_that_holds_on_every_fold(tmp_path: Path) -> None:
    """Лидер, чьё превосходство не держится на разбиении, чемпионом не объявляется."""
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            #выигрывает в среднем, но на одном фолде проваливается ниже точки отсчёта
            "wobbly": {"folds": [0.80, 0.80, 0.30], "label": "Неустойчивая"},
        },
    )
    board = build_leaderboard(
        record=record, store=store, objective=default_objective("binary")
    )

    assert board.champion.contender_key is None
    assert "не держится" in board.champion.reason
    #лидер таблицы при этом виден: «первый в таблице» и «победитель» — разные утверждения
    assert board.rows[0].contender_key == "wobbly"
    assert board.rows[0].rank == 1


def test_champion_is_declared_when_superiority_holds(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            "steady": {"folds": [0.90, 0.90, 0.90], "label": "Устойчивая"},
        },
    )
    board = build_leaderboard(
        record=record, store=store, objective=default_objective("binary")
    )

    assert board.champion.contender_key == "steady"
    assert board.champion.over_baseline["stable"] is True
    assert "на всех 3 фолдах" in board.champion.reason


def test_no_champion_when_baseline_wins(tmp_path: Path) -> None:
    """Если признаки ничего не дали, победителя нет — и это главный вывод прогона."""
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.60, 0.60, 0.60], "is_baseline": True},
            "weak": {"folds": [0.50, 0.50, 0.50], "label": "Слабая"},
        },
    )
    board = build_leaderboard(
        record=record, store=store, objective=default_objective("binary")
    )

    assert board.champion.contender_key is None
    assert "не превзошла постоянного ответа" in board.champion.reason


def test_indistinguishable_runner_up_is_named_as_such(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            "alpha": {"folds": [0.900, 0.900, 0.900], "label": "Альфа", "cost": 5.0},
            "beta": {"folds": [0.900, 0.900, 0.900], "label": "Бета", "cost": 1.0},
        },
    )
    board = build_leaderboard(
        record=record, store=store, objective=default_objective("binary")
    )

    assert board.champion.contender_key is not None
    assert "неотличим" in board.champion.reason
    #порядок определён объявленным правилом, и правило названо
    assert board.champion.decided_by_tie_breaker
    #при равном результате впереди более дешёвая модель
    assert board.champion.contender_key == "beta"


def test_changing_the_objective_reorders_without_retraining(tmp_path: Path) -> None:
    """Смена цели меняет порядок и не требует ни одного нового обучения (D-6)."""
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            "wide": {"folds": [0.85, 0.85, 0.85], "label": "Широкая"},
            "narrow": {"folds": [0.70, 0.70, 0.70], "label": "Узкая"},
        },
    )
    by_accuracy = build_leaderboard(
        record=record, store=store, objective=build_objective("binary", metric="accuracy")
    )
    by_log_loss = build_leaderboard(
        record=record, store=store, objective=build_objective("binary", metric="log_loss")
    )

    assert by_accuracy.rows[0].contender_key == "wide"
    #log_loss — метрика «меньше лучше»: порядок обязан считаться по другому знаку
    assert by_log_loss.objective["higher_is_better"] is False
    assert by_log_loss.rows[0].score is not None
    assert by_log_loss.rows[0].score <= by_log_loss.rows[1].score


def test_holdout_is_shown_but_never_ranked(tmp_path: Path) -> None:
    """Ранжирование по holdout превратило бы независимую проверку в часть выбора."""
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            #на кросс-валидации слабее, но на holdout сильно лучше
            "cv_weak": {"folds": [0.70, 0.70, 0.70], "holdout": 0.99, "label": "Слабая на CV"},
            "cv_strong": {"folds": [0.90, 0.90, 0.90], "holdout": 0.55, "label": "Сильная на CV"},
        },
    )
    board = build_leaderboard(
        record=record, store=store, objective=build_objective("binary", metric="accuracy")
    )

    assert board.rows[0].contender_key == "cv_strong"
    #holdout при этом посчитан и виден
    assert board.rows[0].holdout
    assert any("holdout" in note.lower() for note in board.notes)


def test_contender_without_the_metric_is_not_last_place_with_zero(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            "good": {"folds": [0.90, 0.90, 0.90], "label": "Рабочая"},
            "broken": {
                "folds": [],
                "state": "FAILED",
                "error_message": "Обучение прервалось.",
                "label": "Упавшая",
            },
        },
    )
    board = build_leaderboard(
        record=record, store=store, objective=default_objective("binary")
    )
    broken = next(row for row in board.rows if row.contender_key == "broken")

    assert broken.rank is None
    assert broken.score is None
    assert not broken.eligible
    assert "прервалось" in broken.reason


def test_succeeded_contender_without_the_target_metric_is_not_ranked(tmp_path: Path) -> None:
    """Обучился, но целевую метрику посчитать нечем — это не ноль и не последнее место.

    Модель без вероятностей встречается постоянно, и ранговая метрика для неё
    не определена. Поставить ей ноль значило бы объявить её худшей из всех, хотя
    по ней просто не измеряли то, что выбрано целью.
    """
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            "no_proba": {"folds": [0.90, 0.90, 0.90], "label": "Без вероятностей", "with_proba": False},
        },
    )
    board = build_leaderboard(
        record=record, store=store, objective=build_objective("binary", metric="roc_auc")
    )
    row = next(item for item in board.rows if item.contender_key == "no_proba")

    assert row.score is None
    assert row.rank is None
    assert not row.eligible
    assert "не посчитана" in row.reason
    assert "вероятност" in row.reason
    #по метрике, которая считается, участник при этом виден
    assert any(item["key"] == "accuracy" and item["value"] is not None for item in row.cross_validated)


def test_constraint_removes_from_contention_but_keeps_the_row(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            "greedy": {"folds": [0.95, 0.95, 0.95], "label": "Жадная"},
        },
    )
    #порог заведомо недостижим для построенных предсказаний
    board = build_leaderboard(
        record=record,
        store=store,
        objective=build_objective(
            "binary",
            metric="accuracy",
            constraints=[
                {"metric": "recall", "operator": "gte", "value": 0.999, "reason": "пропуск недопустим"}
            ],
        ),
    )
    greedy = next(row for row in board.rows if row.contender_key == "greedy")

    assert not greedy.eligible
    assert "Не выполнены условия" in greedy.reason
    #строка не исчезла: исчезнув, участник читался бы как проигравший
    assert greedy.score is not None
    assert greedy.constraints[0].satisfied is False


def test_missing_predictions_are_named_not_treated_as_a_bad_result(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path)
    record = _run_with(
        store,
        {
            "baseline_majority": {"folds": [0.50, 0.50, 0.50], "is_baseline": True},
            "gone": {"folds": [0.90, 0.90, 0.90], "label": "Пропавшая"},
        },
    )
    store.predictions_path(record.run_id, "gone").unlink()

    board = build_leaderboard(
        record=record, store=store, objective=default_objective("binary")
    )
    gone = next(row for row in board.rows if row.contender_key == "gone")

    assert gone.rank is None
    assert "не читаются" in gone.reason
