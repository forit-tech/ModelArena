"""Разбиение на фолды: инварианты, которые обязаны выполняться для каждой стратегии."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backend.protocol.folds import _assignment_hash, build_folds
from backend.protocol.recommend import recommend_protocol
from backend.tasks.spec import build_task_spec

ROWS = 300


def frame_for(strategy: str) -> pl.DataFrame:
    if strategy == "time":
        return pl.DataFrame(
            {
                "created": [f"2024-{1 + value // 26:02d}-{1 + value % 26:02d}" for value in range(ROWS)],
                "spend": [round(value * 1.7, 2) for value in range(ROWS)],
                "churned": [value % 4 == 0 for value in range(ROWS)],
            }
        )

    if strategy == "group":
        return pl.DataFrame(
            {
                "customer_id": [value // 5 for value in range(ROWS)],
                "spend": [round(value * 1.7, 2) for value in range(ROWS)],
                "churned": [(value // 5) % 4 == 0 for value in range(ROWS)],
            }
        )

    return pl.DataFrame(
        {
            "spend": [round(value * 1.7, 2) for value in range(ROWS)],
            "churned": [value % 4 == 0 for value in range(ROWS)],
        }
    )


def plan_for(strategy: str):
    frame = frame_for(strategy)
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend"],
        positive_label="True",
        group_column="customer_id" if strategy == "group" else None,
        time_column="created" if strategy == "time" else None,
    )
    protocol = recommend_protocol(frame, spec, n_splits=3)
    return frame, spec, protocol, build_folds(frame, spec, protocol)


@pytest.mark.parametrize("strategy", ["random", "group", "time"])
def test_train_and_validation_never_intersect(strategy: str) -> None:
    _, _, _, folds = plan_for(strategy)

    for index in range(folds.n_splits):
        training = set(folds.training_for(index).tolist())
        validation = set(folds.validation_folds[index].tolist())
        assert not training & validation, f"{strategy}: фолд {index} пересекается"


@pytest.mark.parametrize("strategy", ["random", "group", "time"])
def test_holdout_never_touches_the_training_pool(strategy: str) -> None:
    _, _, _, folds = plan_for(strategy)

    assert not set(folds.holdout.tolist()) & set(folds.train_pool.tolist())


def test_rolling_window_training_is_only_the_past() -> None:
    """Обучение при скользящем окне — только прошлое, а не «пул минус проверка».

    Именно этот инвариант был нарушен: обучающая часть выводилась вычитанием,
    и модель получала будущее.
    """
    frame, _, _, folds = plan_for("time")
    values = frame["created"].str.to_datetime(strict=False).to_numpy()

    for index in range(folds.n_splits):
        training = folds.training_for(index)
        validation = folds.validation_folds[index]
        assert np.nanmax(values[training]) <= np.nanmin(values[validation]), (
            f"фолд {index}: обучение содержит наблюдения позже проверки"
        )


def test_group_isolation_holds_for_group_protocol() -> None:
    frame, _, _, folds = plan_for("group")
    groups = frame["customer_id"].cast(pl.String).to_numpy()

    for index in range(folds.n_splits):
        training = set(groups[folds.training_for(index)].tolist())
        validation = set(groups[folds.validation_folds[index]].tolist())
        assert not training & validation


def test_rows_with_unknown_time_go_to_the_past_not_to_holdout() -> None:
    """Неизвестное время трактуется как прошлое.

    numpy сортирует NaT в конец, и такая строка оказывалась «самой свежей» — уходила
    в holdout, то есть модель проверялась на наблюдении, о котором неизвестно, когда оно было.
    """
    rows = 200
    frame = pl.DataFrame(
        {
            "created": [
                None if value < 3 else f"2024-{1 + value // 26:02d}-{1 + value % 26:02d}"
                for value in range(rows)
            ],
            "spend": [round(value * 1.7, 2) for value in range(rows)],
            "churned": [value % 4 == 0 for value in range(rows)],
        }
    )
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend"],
        positive_label="True",
        time_column="created",
    )
    protocol = recommend_protocol(frame, spec, n_splits=3)
    folds = build_folds(frame, spec, protocol)

    assert not set(range(3)) & set(folds.holdout.tolist())


# ---------------------------------------------------------------- отпечаток разбиения


def test_hash_changes_when_training_changes_even_if_validation_is_equal() -> None:
    """Отпечаток обязан покрывать обучающую часть.

    Хеш только по валидации назвал бы одинаковыми два протокола, различающиеся тем,
    какое прошлое видит модель, — а это разные протоколы.
    """
    holdout = np.array([0, 1])
    validation = np.array([8, 9])

    narrow = _assignment_hash(holdout, [(np.array([2, 3]), validation)])
    wide = _assignment_hash(holdout, [(np.array([2, 3, 4, 5]), validation)])

    assert narrow != wide


def test_hash_is_stable_for_identical_assignment() -> None:
    folds = [(np.array([2, 3]), np.array([4, 5])), (np.array([4, 5]), np.array([2, 3]))]

    assert _assignment_hash(np.array([0, 1]), folds) == _assignment_hash(np.array([0, 1]), folds)


def test_hash_distinguishes_ambiguous_groupings() -> None:
    """Классическая коллизия склейки: [[1],[2,3]] против [[1,2],[3]]."""
    holdout = np.array([0])
    first = [(np.array([9]), np.array([1])), (np.array([9]), np.array([2, 3]))]
    second = [(np.array([9]), np.array([1, 2])), (np.array([9]), np.array([3]))]

    assert _assignment_hash(holdout, first) != _assignment_hash(holdout, second)
