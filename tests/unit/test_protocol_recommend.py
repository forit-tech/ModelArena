"""Рекомендация протокола: выбор стратегии, выполнимость и предупреждения о структуре."""
from __future__ import annotations

import polars as pl
import pytest

from backend.core.errors import ValidationError
from backend.protocol.recommend import recommend_protocol
from backend.tasks.spec import TaskSpec, build_task_spec


def _spec(frame: pl.DataFrame, **overrides: object) -> TaskSpec:
    target = str(overrides.pop("target_column", "churned"))
    features = [
        name
        for name in frame.columns
        if name != target and name != overrides.get("group_column") and name != overrides.get("time_column")
    ]
    return build_task_spec(
        frame,
        target_column=target,
        task_type="binary",
        feature_columns=features,
        positive_label="True",
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture
def balanced() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "spend": [float(value) for value in range(200)],
            "plan": ["basic", "pro"] * 100,
            "churned": [True, False] * 100,
        }
    )


# ---------------------------------------------------------------- классификация


def test_stratified_by_default_and_explains_with_numbers(balanced: pl.DataFrame) -> None:
    proposal = recommend_protocol(balanced, _spec(balanced))

    assert proposal.cv_splitter == "stratified_k_fold"
    assert proposal.holdout_strategy == "stratified"
    assert proposal.n_splits == 5
    #объяснение обязано содержать числа, а не общие слова про «сохранение долей»
    assert "100" in proposal.explanation


def test_rare_class_reduces_folds_and_says_why() -> None:
    #три объекта редкого класса не разложить на пять фолдов; понижение обязано быть видимым
    frame = pl.DataFrame(
        {
            "spend": [float(value) for value in range(100)],
            "churned": [True] * 3 + [False] * 97,
        }
    )
    proposal = recommend_protocol(frame, _spec(frame))

    assert proposal.n_splits == 3
    assert any("применено 3" in note for note in proposal.feasibility_notes)


def test_class_that_cannot_be_split_is_refused() -> None:
    frame = pl.DataFrame(
        {"spend": [float(value) for value in range(60)], "churned": [True] + [False] * 59}
    )

    with pytest.raises(ValidationError, match="встречается 1 раз"):
        recommend_protocol(frame, _spec(frame))


def test_thin_fold_is_reported_even_when_feasible() -> None:
    #формально пять фолдов возможны, но по одному объекту редкого класса на фолд —
    #это уже не измерение, и молчать об этом нельзя
    frame = pl.DataFrame(
        {
            "spend": [float(value) for value in range(300)],
            "churned": [True] * 6 + [False] * 294,
        }
    )
    proposal = recommend_protocol(frame, _spec(frame))

    assert proposal.n_splits == 5
    assert any("прыгать между фолдами" in note for note in proposal.feasibility_notes)


# ---------------------------------------------------------------- регрессия


def test_regression_uses_plain_k_fold() -> None:
    frame = pl.DataFrame(
        {
            "feature": [float(value) for value in range(200)],
            "price": [float(value) for value in range(200)],
        }
    )
    spec = build_task_spec(
        frame, target_column="price", task_type="regression", feature_columns=["feature"]
    )
    proposal = recommend_protocol(frame, spec)

    assert proposal.cv_splitter == "k_fold"
    assert proposal.holdout_strategy == "random"


# ---------------------------------------------------------------- группы


def test_group_column_switches_strategy_and_keeps_entities_whole() -> None:
    frame = pl.DataFrame(
        {
            "customer_id": [value // 5 for value in range(200)],   # 40 сущностей по 5 строк
            "spend": [float(value) for value in range(200)],
            "churned": [True, False] * 100,
        }
    )
    proposal = recommend_protocol(frame, _spec(frame, group_column="customer_id"))

    assert proposal.cv_splitter in {"group_k_fold", "stratified_group_k_fold"}
    assert proposal.holdout_strategy == "group"
    assert "узнаёт сущность" in proposal.explanation


def test_too_few_groups_is_refused() -> None:
    frame = pl.DataFrame(
        {
            "customer_id": [0] * 100,
            "spend": [float(value) for value in range(100)],
            "churned": [True, False] * 50,
        }
    )

    with pytest.raises(ValidationError, match="сущност"):
        recommend_protocol(frame, _spec(frame, group_column="customer_id"))


# ---------------------------------------------------------------- время


def test_time_column_switches_to_forward_chaining() -> None:
    frame = pl.DataFrame(
        {
            "created": [f"2024-{month:02d}-01" for month in range(1, 13)] * 10,
            "spend": [float(value) for value in range(120)],
            "churned": [True, False] * 60,
        }
    )
    proposal = recommend_protocol(frame, _spec(frame, time_column="created"))

    assert proposal.cv_splitter == "forward_chaining"
    assert proposal.holdout_strategy == "temporal"
    #перемешивание уничтожило бы порядок, ради которого выбрана стратегия
    assert proposal.shuffle is False
    assert "из будущего в прошлое" in proposal.explanation


def test_too_few_periods_falls_back_but_keeps_temporal_holdout() -> None:
    frame = pl.DataFrame(
        {
            "created": ["2024-01-01", "2024-02-01"] * 50,
            "spend": [float(value) for value in range(100)],
            "churned": [True, False] * 50,
        }
    )
    proposal = recommend_protocol(frame, _spec(frame, time_column="created"))

    assert proposal.cv_splitter == "k_fold"
    #holdout всё равно берётся с конца: обучать на будущем и проверять на прошлом нельзя
    assert proposal.holdout_strategy == "temporal"
    assert any("различных момента" in note for note in proposal.feasibility_notes)


# ---------------------------------------------------------------- структура, которую не выбрали


def test_unselected_group_candidate_is_reported(balanced: pl.DataFrame) -> None:
    #самая дорогая ошибка — не заметить структуру, которая в данных есть
    frame = balanced.with_columns(
        pl.Series("customer_id", [value // 5 for value in range(balanced.height)])
    )
    proposal = recommend_protocol(frame, _spec(frame))

    assert any("идентификатор сущности" in warning for warning in proposal.warnings)


def test_unselected_time_candidate_is_reported() -> None:
    frame = pl.DataFrame(
        {
            "created": [f"2024-{month:02d}-01" for month in range(1, 13)] * 10,
            "spend": [float(value) for value in range(120)],
            "churned": [True, False] * 60,
        }
    )
    proposal = recommend_protocol(frame, _spec(frame))

    assert any("утечку" in warning for warning in proposal.warnings)


# ---------------------------------------------------------------- размеры и настройки


def test_small_dataset_and_thin_holdout_are_reported() -> None:
    frame = pl.DataFrame(
        {"spend": [float(value) for value in range(40)], "churned": [True, False] * 20}
    )
    proposal = recommend_protocol(frame, _spec(frame))

    assert any("небольшой" in warning for warning in proposal.warnings)
    assert any("Доверительные интервалы" in warning for warning in proposal.warnings)


def test_invalid_settings_are_refused(balanced: pl.DataFrame) -> None:
    spec = _spec(balanced)

    with pytest.raises(ValidationError, match="фолдов"):
        recommend_protocol(balanced, spec, n_splits=1)

    with pytest.raises(ValidationError, match="holdout"):
        recommend_protocol(balanced, spec, holdout_size=0.9)


def test_null_targets_do_not_participate() -> None:
    #строки без цели не должны влиять ни на выбор стратегии, ни на подсчёт редкого класса
    frame = pl.DataFrame(
        {
            "spend": [float(value) for value in range(100)],
            "churned": [True, False] * 25 + [None] * 50,
        }
    )
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend"],
        positive_label="True",
    )
    proposal = recommend_protocol(frame, spec)

    assert proposal.cv_splitter == "stratified_k_fold"
    assert "из 50" in proposal.explanation
