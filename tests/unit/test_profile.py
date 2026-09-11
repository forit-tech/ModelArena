"""Профиль снимка: проверяемые величины вместо одного числа."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from backend.datasets.package import read_package
from backend.datasets.profile import build_profile

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "dataset-package" / "v1"


def test_counts_missing_duplicates_and_uniqueness() -> None:
    frame = pl.DataFrame(
        {
            "value": [1, 1, 2, None],
            "label": ["a", "a", "b", "b"],
        }
    )
    profile = build_profile(frame)

    assert profile.row_count == 4
    assert profile.missing_cell_count == 1
    assert profile.missing_cell_ratio == 0.125
    #первая и вторая строки совпадают целиком — именно такие дубли завышают метрику,
    #попав одновременно в обучение и в проверку
    assert profile.duplicate_row_count == 1
    assert profile.column("value").missing_count == 1


def test_autoincrement_is_recognised_as_identifier() -> None:
    frame = pl.DataFrame(
        {"id": list(range(1, 21)), "value": [float(value) for value in range(20)]}
    )
    identifier = build_profile(frame).column("id")

    assert identifier.is_probable_id
    #причина обязана быть содержательной: «похоже на ID» ничего не объясняет
    assert "автоинкремент" in (identifier.id_reason or "")


def test_unique_float_is_a_measurement_not_a_key() -> None:
    #уникальное дробное значение в каждой строке — это измерение, а не ключ
    frame = pl.DataFrame(
        {"weight": [value + 0.5 for value in range(30)], "label": ["a", "b"] * 15}
    )

    assert not build_profile(frame).column("weight").is_probable_id


def test_id_detection_is_disabled_on_short_tables() -> None:
    #на пяти строках почти всё уникально, и относительный порог назвал бы идентификатором
    #обычный признак — тот же класс ошибки, что был в пороге типа задачи
    frame = pl.DataFrame({"value": [10, 20, 30, 40, 50], "label": ["a", "b", "a", "b", "a"]})

    assert not build_profile(frame).column("value").is_probable_id


def test_constant_and_near_constant_are_distinguished() -> None:
    frame = pl.DataFrame(
        {
            "constant": ["same"] * 200,
            "near_constant": ["same"] * 199 + ["other"],
            "varied": ["a", "b"] * 100,
        }
    )
    profile = build_profile(frame)

    assert profile.column("constant").is_constant
    #почти-константа не является константой, но различать объекты тоже не помогает
    assert not profile.column("near_constant").is_constant
    assert profile.column("near_constant").is_near_constant
    assert not profile.column("varied").is_near_constant


def test_numeric_stats_are_present_for_numbers_only() -> None:
    frame = pl.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0], "label": ["a", "b", "a", "b"]})
    profile = build_profile(frame)

    stats = profile.column("amount").numeric_stats
    assert stats is not None
    assert stats["median"] == 2.5
    assert profile.column("label").numeric_stats is None
    assert profile.column("label").top_values is not None


def test_nulls_do_not_leak_into_statistics() -> None:
    frame = pl.DataFrame({"amount": [1.0, None, 3.0], "label": ["a", "b", "a"]})
    stats = build_profile(frame).column("amount").numeric_stats

    assert stats is not None
    #пропуск — это отсутствие значения, а не ноль: подстановка нуля сместила бы среднее
    assert stats["mean"] == 2.0


def test_profile_of_golden_package_reads() -> None:
    loaded = read_package(
        CONTRACT_ROOT / "golden" / "customers_golden.dapkg", CONTRACT_ROOT / "schema"
    )
    profile = build_profile(loaded.frame)

    assert profile.row_count == 12
    assert profile.column("customer_id").is_probable_id
    assert profile.column("churned").top_values is not None
