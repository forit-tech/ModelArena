"""Постановка задачи: определение типа, предложение признаков, проверки."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from backend.core.errors import ValidationError
from backend.datasets.registry import ROW_ID, DatasetRegistry
from backend.tasks.inference import (
    find_group_column_candidates,
    find_time_column_candidates,
    infer_task_type,
)
from backend.tasks.spec import ExclusionRecord, build_task_spec, propose_task_setup

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "dataset-package" / "v1"
GOLDEN_PACKAGE = CONTRACT_ROOT / "golden" / "customers_golden.dapkg"


# ---------------------------------------------------------------- тип задачи


def test_two_values_give_binary_and_rare_class_is_positive() -> None:
    #по умолчанию интересен редкий класс: отток, дефолт, брак
    series = pl.Series("churned", [False] * 18 + [True] * 2)
    proposal = infer_task_type(series)

    assert proposal.task_type == "binary"
    assert proposal.positive_label == "True"
    assert "реже" in proposal.explanation


def test_few_distinct_numbers_are_classes_not_a_quantity() -> None:
    series = pl.Series("rating", [1, 2, 3, 4, 5] * 40)
    assert infer_task_type(series).task_type == "multiclass"


def test_many_distinct_numbers_are_regression() -> None:
    series = pl.Series("price", [float(value) for value in range(200)])
    proposal = infer_task_type(series)

    assert proposal.task_type == "regression"
    assert proposal.class_labels == []


def test_threshold_has_a_floor_so_short_datasets_are_not_mislabelled() -> None:
    #20 классов на 100 строк — это классификация. Без нижней границы доля 5% дала бы
    #порог в 5 классов, и распознавание цифр по сотне примеров стало бы «регрессией»
    assert infer_task_type(pl.Series("value", list(range(20)) * 5)).task_type == "multiclass"
    assert infer_task_type(pl.Series("value", list(range(20)) * 200)).task_type == "multiclass"
    #а 400 различных значений на 400 строк — величина, а не классы
    assert infer_task_type(pl.Series("value", list(range(400)))).task_type == "regression"


def test_constant_target_is_rejected_with_a_reason() -> None:
    with pytest.raises(ValidationError, match="одно значение"):
        infer_task_type(pl.Series("flag", ["yes"] * 50))


def test_empty_target_is_rejected() -> None:
    with pytest.raises(ValidationError, match="пуста"):
        infer_task_type(pl.Series("target", [None, None], dtype=pl.String))


def test_datetime_target_is_refused_explicitly() -> None:
    #честнее отказать, чем сделать вид, что это регрессия
    series = pl.Series("signup_date", ["2024-01-01", "2024-02-01"]).str.to_date()

    with pytest.raises(ValidationError, match="временных рядов"):
        infer_task_type(series)


def test_null_targets_are_reported_not_hidden() -> None:
    series = pl.Series("churned", [True, False, None, True, False])
    assert any("не участвуют" in warning for warning in infer_task_type(series).warnings)


# ---------------------------------------------------------------- кандидаты


def test_time_candidates_are_found_by_values_not_by_name() -> None:
    frame = pl.DataFrame(
        {
            "runtime": list(range(30)),                       # имя намекает, содержимое нет
            "created": [f"2024-01-{day:02d}" for day in range(1, 31)],
        }
    )

    assert find_time_column_candidates(frame) == ["created"]


def test_group_candidates_require_repetition() -> None:
    frame = pl.DataFrame(
        {
            "customer_id": [value // 4 for value in range(40)],   # 10 сущностей по 4 строки
            "unique_id": list(range(40)),                         # уникален — не группа
            "value": [float(value) for value in range(40)],
        }
    )

    candidates = find_group_column_candidates(frame)

    assert "customer_id" in candidates
    assert "unique_id" not in candidates


# ---------------------------------------------------------------- предложение


def test_golden_package_is_too_small_and_says_so(tmp_path: Path) -> None:
    #эталонный пакет контракта — двенадцать строк, и у одной цель пуста: он собран
    #для проверки формата, а не для обучения. Отказ обязан называть причину числом
    registry = DatasetRegistry(root=tmp_path / "datasets")
    snapshot = registry.import_package(GOLDEN_PACKAGE)

    with pytest.raises(ValidationError, match="11"):
        propose_task_setup(snapshot, registry.load_frame(snapshot.dataset_id), "churned")


def test_proposal_excludes_row_key_with_a_reason(tmp_path: Path) -> None:
    registry = DatasetRegistry(root=tmp_path / "datasets")
    path = tmp_path / "customers.parquet"
    pl.DataFrame(
        {
            "customer_id": list(range(40)),
            "spend": [float(value) for value in range(40)],
            "plan": ["basic", "pro"] * 20,
            "churned": [True, False] * 20,
        }
    ).write_parquet(path)
    stored = registry.import_file(path)
    #снимок из файла не несёт подсказок пакета, поэтому подставляем их явно:
    #проверяется реакция на подсказку, а не путь импорта
    snapshot = replace(
        stored,
        row_key=["customer_id"],
        package_ref={"hints": {"identifier_columns": ["customer_id"]}},
    )

    proposal = propose_task_setup(snapshot, registry.load_frame(stored.dataset_id), "churned")

    assert proposal.task_type == "binary"
    assert "customer_id" not in proposal.feature_columns
    excluded = {record.column: record for record in proposal.suggested_exclusions}
    assert excluded["customer_id"].source == "package_hint"
    #причина обязана быть содержательной: через месяц вопрос «почему исключили» возникнет
    assert "row_key" in excluded["customer_id"].reason


def test_proposal_refuses_tiny_dataset(tmp_path: Path) -> None:
    registry = DatasetRegistry(root=tmp_path / "datasets")
    path = tmp_path / "tiny.parquet"
    pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).write_parquet(path)
    snapshot = registry.import_file(path)

    with pytest.raises(ValidationError, match="меньше"):
        propose_task_setup(snapshot, registry.load_frame(snapshot.dataset_id), "y")


# ---------------------------------------------------------------- сборка спецификации


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": list(range(30)),
            "spend": [float(value) for value in range(30)],
            "plan": ["basic", "pro"] * 15,
            "churned": [True, False] * 15,
        }
    )


def test_build_spec_fixes_label_order(frame: pl.DataFrame) -> None:
    #порядок меток фиксируется на весь прогон: по нему выравниваются вероятности всех моделей
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend", "plan"],
        positive_label="True",
    )

    assert spec.class_labels == ["False", "True"]
    assert spec.positive_label == "True"


def test_binary_without_positive_label_is_refused(frame: pl.DataFrame) -> None:
    with pytest.raises(ValidationError, match="положительный класс"):
        build_task_spec(
            frame,
            target_column="churned",
            task_type="binary",
            feature_columns=["spend"],
        )


def test_unknown_positive_label_is_refused(frame: pl.DataFrame) -> None:
    with pytest.raises(ValidationError, match="отсутствует"):
        build_task_spec(
            frame,
            target_column="churned",
            task_type="binary",
            feature_columns=["spend"],
            positive_label="maybe",
        )


def test_target_cannot_be_a_feature(frame: pl.DataFrame) -> None:
    with pytest.raises(ValidationError, match="одновременно"):
        build_task_spec(
            frame,
            target_column="churned",
            task_type="binary",
            feature_columns=["spend", "churned"],
            positive_label="True",
        )


def test_row_id_cannot_be_a_feature(frame: pl.DataFrame) -> None:
    #инвариант I-8: служебный идентификатор не попадает в матрицу признаков никогда
    with_ids = frame.with_row_index(name=ROW_ID)

    with pytest.raises(ValidationError, match=ROW_ID):
        build_task_spec(
            with_ids,
            target_column="churned",
            task_type="binary",
            feature_columns=["spend", ROW_ID],
            positive_label="True",
        )


def test_missing_feature_is_named(frame: pl.DataFrame) -> None:
    with pytest.raises(ValidationError, match="nonexistent"):
        build_task_spec(
            frame,
            target_column="churned",
            task_type="binary",
            feature_columns=["spend", "nonexistent"],
            positive_label="True",
        )


def test_exclusions_keep_their_reason(frame: pl.DataFrame) -> None:
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend", "plan"],
        excluded_columns=[
            ExclusionRecord(column="customer_id", reason="идентификатор", source="user")
        ],
        positive_label="True",
    )

    assert spec.excluded_columns[0].reason == "идентификатор"
    assert spec.to_dict()["excluded_columns"][0]["source"] == "user"
