"""Реестр снимков: неизменяемость, идентичность по содержимому, служебные колонки."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from backend.core.errors import NotFoundError, ValidationError
from backend.datasets.registry import ROW_ID, DatasetRegistry

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "dataset-package" / "v1"
GOLDEN_PACKAGE = CONTRACT_ROOT / "golden" / "customers_golden.dapkg"


@pytest.fixture
def registry(tmp_path: Path) -> DatasetRegistry:
    return DatasetRegistry(root=tmp_path / "datasets")


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "spend": [10.5, 20.0, 15.25, 30.0, 12.0, 18.5],
            "plan": ["basic", "pro", "basic", "pro", "basic", "pro"],
            "churned": [False, True, False, True, False, True],
        }
    )


def test_package_import_creates_snapshot(registry: DatasetRegistry) -> None:
    snapshot = registry.import_package(GOLDEN_PACKAGE)

    assert snapshot.source == "package"
    assert snapshot.row_count == 12
    assert snapshot.fingerprint_algorithm == "dataarena-logical-sha256-v1"
    #отпечаток снимка — тот же, что объявлен в манифесте: идентичность не переизобретается
    assert snapshot.fingerprint == "457184a0a0dc03776d8e8557ab0e02092f09b67866fa453fdfd51615116adac3"
    assert snapshot.row_key == ["customer_id"]


def test_reimport_of_same_data_returns_same_snapshot(registry: DatasetRegistry) -> None:
    #на снимок ссылаются эксперименты; повторный импорт не должен создавать дубль
    first = registry.import_package(GOLDEN_PACKAGE)
    second = registry.import_package(GOLDEN_PACKAGE)

    assert first.dataset_id == second.dataset_id
    assert len(registry.list_snapshots()) == 1


def test_csv_and_parquet_with_same_content_give_one_snapshot(
    registry: DatasetRegistry,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    #отпечаток логический: он описывает данные, а не способ их записи (D-14)
    parquet_path = tmp_path / "sample.parquet"
    csv_path = tmp_path / "sample.csv"
    sample_frame.write_parquet(parquet_path)
    sample_frame.write_csv(csv_path)

    from_parquet = registry.import_file(parquet_path)
    from_csv = registry.import_file(csv_path)

    assert from_parquet.fingerprint == from_csv.fingerprint
    assert from_parquet.dataset_id == from_csv.dataset_id


def test_changed_value_changes_identity(
    registry: DatasetRegistry,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "a.parquet"
    second_path = tmp_path / "b.parquet"
    sample_frame.write_parquet(first_path)
    sample_frame.with_columns(pl.col("spend") * 2).write_parquet(second_path)

    assert registry.import_file(first_path).fingerprint != registry.import_file(second_path).fingerprint


def test_snapshot_data_is_readable_and_row_id_is_opt_in(
    registry: DatasetRegistry,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.parquet"
    sample_frame.write_parquet(path)
    snapshot = registry.import_file(path)

    plain = registry.load_frame(snapshot.dataset_id)
    with_ids = registry.load_frame(snapshot.dataset_id, with_row_id=True)

    #служебный идентификатор появляется только по явному запросу и никогда сам собой
    assert ROW_ID not in plain.columns
    assert with_ids[ROW_ID].to_list() == list(range(sample_frame.height))


def test_reserved_column_is_rejected(registry: DatasetRegistry, tmp_path: Path) -> None:
    path = tmp_path / "reserved.parquet"
    pl.DataFrame({ROW_ID: [1, 2, 3], "value": [1, 2, 3]}).write_parquet(path)

    with pytest.raises(ValidationError, match=ROW_ID):
        registry.import_file(path)


def test_single_column_dataset_is_rejected(registry: DatasetRegistry, tmp_path: Path) -> None:
    path = tmp_path / "thin.parquet"
    pl.DataFrame({"only": [1, 2, 3]}).write_parquet(path)

    with pytest.raises(ValidationError, match="признак"):
        registry.import_file(path)


def test_unknown_dataset_id_is_reported(registry: DatasetRegistry) -> None:
    with pytest.raises(NotFoundError):
        registry.get("ds_0000000000000000000000")


def test_malformed_dataset_id_does_not_reach_filesystem(registry: DatasetRegistry) -> None:
    #идентификатор попадает в путь, поэтому произвольная строка сюда не проходит
    with pytest.raises(ValidationError):
        registry.get("../../etc")
