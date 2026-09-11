"""Регрессионный тест контракта Dataset Package.

Проверяется продакшн-реализация `backend.datasets.package`, написанная по тексту
спецификации и не импортирующая эталонные `reference/*.py` (D-15). Матрица ожиданий —
из `contracts/dataset-package/v1/README.md`.

Фикстура, появившаяся в каталоге и не описанная здесь, падает как FAIL, а не проходит
молча: иначе пополнение набора владельцем прошло бы мимо проверки потребителя.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from backend.core.errors import PackageError
from backend.datasets.package import content_fingerprint, read_package

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "dataset-package" / "v1"
SCHEMA_DIRECTORY = CONTRACT_ROOT / "schema"

#(принят, код отказа, ожидается предупреждение)
EXPECTATIONS: dict[str, tuple[bool, str | None, bool]] = {
    "golden/customers_golden.dapkg": (True, None, False),
    "golden/customers_golden.dapkg.zip": (True, None, False),
    "golden/negative/minimal.dapkg": (True, None, False),
    "golden/negative/future-minor.dapkg": (True, None, True),
    "golden/negative/sampled-profile.dapkg": (True, None, True),
    "golden/negative/future-major.dapkg": (False, "package_version_unsupported", False),
    "golden/negative/corrupted-part.dapkg": (False, "package_integrity_failed", False),
    "golden/negative/fingerprint-mismatch.dapkg": (False, "package_fingerprint_mismatch", False),
    "golden/negative/malformed-schema.dapkg": (False, "package_part_invalid", False),
    "golden/negative/missing-artifact.dapkg": (False, "package_part_missing", False),
    "golden/negative/duplicate-entry.dapkg": (False, "package_duplicate_entry", False),
    "golden/negative/schema-mismatch.dapkg": (False, "package_schema_mismatch", False),
    "golden/negative/traversal.dapkg.zip": (False, "package_unsafe_path", False),
    "golden/negative/zip-bomb.dapkg.zip": (False, "package_limit_exceeded", False),
}


def test_contract_mirror_present() -> None:
    #без зеркала весь набор ниже пропустился бы молча
    assert (CONTRACT_ROOT / "dataset-package-v1.md").exists()
    assert (CONTRACT_ROOT / "golden" / "customers_golden.dapkg" / "manifest.json").exists()


def test_no_undescribed_fixtures() -> None:
    known = {name.split("/")[-1] for name in EXPECTATIONS}
    present = {child.name for child in (CONTRACT_ROOT / "golden" / "negative").iterdir()}
    assert present <= known, f"в контракте появились фикстуры без ожидания: {sorted(present - known)}"


@pytest.mark.parametrize("relative", sorted(EXPECTATIONS))
def test_fixture_matches_expectation(relative: str) -> None:
    expect_accepted, expect_code, expect_warning = EXPECTATIONS[relative]

    try:
        loaded = read_package(CONTRACT_ROOT / relative, SCHEMA_DIRECTORY)
    except PackageError as error:
        assert not expect_accepted, f"{relative}: ожидался приём, получен отказ {error.code}"
        assert error.code == expect_code, f"{relative}: код {error.code}, ожидался {expect_code}"
        return

    assert expect_accepted, f"{relative}: ожидался отказ {expect_code}, пакет принят"

    if expect_warning:
        assert loaded.warnings, f"{relative}: ожидалось предупреждение, его нет"


def test_golden_fingerprint_reproduces() -> None:
    #отпечаток — идентичность датасета (D-16): она попадает в карточки экспериментов
    #и живёт там годами, поэтому её воспроизводимость проверяется отдельно от матрицы
    package = CONTRACT_ROOT / "golden" / "customers_golden.dapkg"
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    frame = pl.concat(
        [pl.read_parquet(package / name) for name in manifest["data"]["files"]], how="vertical"
    )

    assert content_fingerprint(frame) == manifest["data"]["content_fingerprint"]["value"]


def test_row_key_is_exposed_but_not_a_feature() -> None:
    #row_key — устойчивая ссылка между сервисами, а не сигнал для модели
    loaded = read_package(CONTRACT_ROOT / "golden" / "customers_golden.dapkg", SCHEMA_DIRECTORY)
    assert loaded.row_key == ["customer_id"]
