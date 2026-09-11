"""Регрессия на дефекты, найденные adversarial-review.

Каждый тест воспроизводит конкретную атаку или сбой, которые **работали** до правки.
Идентификаторы совпадают с SECURITY_AND_ROBUSTNESS_REVIEW.md.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import polars as pl
import pytest

from backend.core.errors import PackageError, ValidationError
from backend.datasets.package import read_package
from backend.datasets.registry import DatasetRegistry
from backend.tasks.inference import MAX_ALLOWED_CLASSES, infer_task_type

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "dataset-package" / "v1"
GOLDEN = CONTRACT_ROOT / "golden" / "customers_golden.dapkg"
SCHEMA_DIR = CONTRACT_ROOT / "schema"


def _clone_with_manifest(tmp_path: Path, mutate) -> Path:
    package = tmp_path / "mutated.dapkg"
    shutil.copytree(GOLDEN, package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    (package / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return package


# ---------------------------------------------------------------- F-01 обход пути


def test_manifest_part_cannot_point_outside_the_package(tmp_path: Path) -> None:
    """F-01. Раньше файл за пределами пакета читался, а совпадение sha256 подтверждалось.

    Это делало ModelArena оракулом по чужой файловой системе: подобрав хеш, отправитель
    пакета мог убедиться в содержимом файла на машине пользователя.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("данные за пределами пакета", encoding="utf-8")

    package = _clone_with_manifest(
        tmp_path,
        lambda manifest: manifest["parts"].append(
            {
                "path": "../secret.txt",
                "bytes": secret.stat().st_size,
                "sha256": hashlib.sha256(secret.read_bytes()).hexdigest(),
            }
        ),
    )

    with pytest.raises(PackageError) as failure:
        read_package(package, SCHEMA_DIR)

    assert failure.value.code == "package_unsafe_path"


def test_data_file_cannot_point_outside_the_package(tmp_path: Path) -> None:
    """F-01. Тот же обход через data.files приводил к чтению произвольного parquet."""
    package = _clone_with_manifest(
        tmp_path, lambda manifest: manifest["data"].update({"files": ["../../elsewhere.parquet"]})
    )

    with pytest.raises(PackageError) as failure:
        read_package(package, SCHEMA_DIR)

    assert failure.value.code == "package_unsafe_path"


def test_absolute_path_in_manifest_is_refused(tmp_path: Path) -> None:
    """F-01. Абсолютный путь обходит проверку «нет .. в пути»."""
    absolute = "C:/Windows/win.ini" if Path("C:/").exists() else "/etc/passwd"
    package = _clone_with_manifest(
        tmp_path, lambda manifest: manifest["data"].update({"files": [absolute]})
    )

    with pytest.raises(PackageError) as failure:
        read_package(package, SCHEMA_DIR)

    assert failure.value.code == "package_unsafe_path"


# ---------------------------------------------------------------- F-02 объём распаковки


def test_uncompressed_archive_over_budget_is_refused(tmp_path: Path) -> None:
    """F-02. Архив без сжатия даёт коэффициент 1:1 и проходил проверку соотношения насквозь.

    Абсолютного предела на распакованный объём не было вообще, хотя настройка
    `max_package_data_bytes` существовала — то есть лимит был объявлен, но не работал.
    """
    archive = tmp_path / "stored.dapkg.zip"

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as sink:
        sink.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "dataarena.package",
                    "format_version": "1.0.0",
                    "data": {"files": ["data/part-0000.parquet"]},
                    "parts": [],
                }
            ),
        )
        sink.writestr("big.bin", b"\x00" * (4 * 1024 * 1024))

    with pytest.raises(PackageError) as failure:
        read_package(archive, SCHEMA_DIR, data_budget_bytes=1024 * 1024)

    assert failure.value.code == "package_limit_exceeded"
    #сообщение обязано называть это ограничением потребителя, а не дефектом пакета
    assert "ограничение ModelArena" in failure.value.message


def test_nested_archive_is_refused(tmp_path: Path) -> None:
    """F-02. Вложенный архив обходит любой предел распаковки рекурсией."""
    inner = tmp_path / "inner.zip"

    with zipfile.ZipFile(inner, "w") as sink:
        sink.writestr("payload.bin", b"0" * 1024)

    archive = tmp_path / "nested.dapkg.zip"

    with zipfile.ZipFile(archive, "w") as sink:
        sink.writestr("manifest.json", "{}")
        sink.write(inner, "inner.zip")

    with pytest.raises(PackageError) as failure:
        read_package(archive, SCHEMA_DIR)

    assert failure.value.code == "package_unsafe_path"


# ---------------------------------------------------------------- F-03 битый манифест


@pytest.mark.parametrize(
    "manifest",
    [
        pytest.param({"format": "dataarena.package"}, id="без format_version"),
        pytest.param(
            {"format": "dataarena.package", "format_version": "abc", "data": {"files": ["a"]}},
            id="версия не semver",
        ),
        pytest.param(
            {"format": "dataarena.package", "format_version": "1.0.0", "data": {"files": []}},
            id="пустой список файлов",
        ),
        pytest.param(
            {"format": "dataarena.package", "format_version": "1.0.0"},
            id="без блока data",
        ),
    ],
)
def test_malformed_manifest_is_a_clean_refusal_not_a_crash(
    tmp_path: Path, manifest: dict
) -> None:
    """F-03, F-04. Раньше это были KeyError и ValueError, то есть пятисотка без объяснения.

    Дефект пакета обязан выглядеть как отказ с кодом, а не как сбой сервиса.
    """
    package = tmp_path / "broken.dapkg"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackageError) as failure:
        read_package(package, SCHEMA_DIR)

    assert failure.value.code in {"package_part_invalid", "package_version_unsupported"}


# ---------------------------------------------------------------- F-05 кардинальность цели


def test_identifier_like_target_is_refused(tmp_path: Path) -> None:
    """F-05. Строковая цель с 200 000 значений принималась как multiclass.

    Список меток целиком уходил в JSON-ответ и осел бы в карточке эксперимента.
    """
    series = pl.Series("email", [f"user{index}@example.com" for index in range(MAX_ALLOWED_CLASSES + 1)])

    with pytest.raises(ValidationError) as failure:
        infer_task_type(series)

    assert "идентификатор или свободный текст" in str(failure.value)


# ---------------------------------------------------------------- F-06 идентификатор снимка


@pytest.mark.parametrize(
    "dataset_id",
    [
        pytest.param("ds_١٢٣", id="арабо-индийские цифры"),
        pytest.param("ds_ＡＢＣ", id="полноширинные латинские"),
        pytest.param("ds_абв", id="кириллица"),
        pytest.param("ds_../x", id="обход пути"),
        pytest.param("../ds_1", id="обход до префикса"),
    ],
)
def test_dataset_id_alphabet_is_narrow(tmp_path: Path, dataset_id: str) -> None:
    """F-06. `str.isalnum()` пропускает Unicode-цифры и буквы.

    Обхода пути это не давало, но на регистронезависимой файловой системе такие
    идентификаторы способны схлопнуться в один каталог.
    """
    registry = DatasetRegistry(root=tmp_path / "datasets")

    with pytest.raises(ValidationError):
        registry.get(dataset_id)


# ---------------------------------------------------------------- F-07 частичное состояние


def test_directory_without_card_does_not_become_an_invisible_duplicate(tmp_path: Path) -> None:
    """F-07. Снимок собирается в staging и переносится одним движением.

    Каталог без `meta.json` — след прерванной записи. Он не должен ни попадать в список,
    ни ломать поиск по отпечатку, иначе следующий импорт тех же данных создаст дубль.
    """
    root = tmp_path / "datasets"
    registry = DatasetRegistry(root=root)
    path = tmp_path / "sample.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).write_parquet(path)
    snapshot = registry.import_file(path)

    #имитируем прерванную запись: каталог есть, карточки нет
    (root / "ds_deadbeefdeadbeefdead01").mkdir()
    (root / ".staging_ds_deadbeefdeadbeefdead02").mkdir()

    listed = registry.list_snapshots()

    assert [item.dataset_id for item in listed] == [snapshot.dataset_id]
    assert registry.find_by_fingerprint(snapshot.fingerprint) is not None


def test_corrupted_card_is_skipped_but_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """F-08. Нечитаемая карточка пропускалась молча — снимок исчезал из поиска бесследно."""
    root = tmp_path / "datasets"
    registry = DatasetRegistry(root=root)
    path = tmp_path / "sample.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).write_parquet(path)
    snapshot = registry.import_file(path)
    (root / snapshot.dataset_id / "meta.json").write_text("{не json", encoding="utf-8")

    with caplog.at_level("WARNING", logger="modelarena.datasets"):
        assert registry.list_snapshots() == []

    assert any("не читается" in record.message for record in caplog.records)


# ---------------------------------------------------------------- F-14 имя загруженного файла


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("../../etc/passwd", "passwd", id="обход пути срезается"),
        pytest.param("..", "dataset", id="две точки не схлопывают путь"),
        pytest.param(".", "dataset", id="точка не схлопывает путь"),
        pytest.param("", "dataset", id="пустое имя"),
        pytest.param(None, "dataset", id="имя отсутствует"),
        pytest.param(r"C:\Windows\win.ini", "win.ini", id="windows-путь"),
        pytest.param("отчёт продаж.csv", "отчёт продаж.csv", id="пробелы и кириллица сохраняются"),
    ],
)
def test_upload_name_edge_cases(raw: str | None, expected: str) -> None:
    """F-14. `Path(name).name` на «.» и «..» даёт пустую строку, и путь схлопывался
    в сам временный каталог — открытие на запись падало пятисоткой."""
    from backend.api.routes.datasets import _safe_upload_name

    assert _safe_upload_name(raw) == expected


def test_upload_name_with_null_byte_is_refused() -> None:
    from backend.api.routes.datasets import _safe_upload_name

    with pytest.raises(ValidationError):
        _safe_upload_name("data\x00.csv")


def test_very_long_upload_name_is_truncated() -> None:
    from backend.api.routes.datasets import _safe_upload_name

    assert len(_safe_upload_name("a" * 4000 + ".csv")) <= 180


# ---------------------------------------------------------------- F-09 понижение строгости


def test_missing_json_schema_downgrade_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-09. Без каталога JSON Schema проверка метаданных пропускается.

    Само по себе это допустимо, но молчать об этом нельзя: пакет с некорректной
    `schema.json` прошёл бы как полностью проверенный, и понижение строгости было бы
    невидимым — ни в карточке снимка, ни в интерфейсе.
    """
    from backend.core.config import get_settings

    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MARENA_CONTRACTS_DIR", str(tmp_path / "нет-такого-каталога"))
    get_settings.cache_clear()

    try:
        registry = DatasetRegistry(root=tmp_path / "datasets")
        snapshot = registry.import_package(GOLDEN)

        assert any(
            "не проверены по схеме" in warning for warning in snapshot.warnings
        ), f"понижение строгости не попало в карточку: {snapshot.warnings}"
    finally:
        get_settings.cache_clear()


def test_present_json_schema_produces_no_downgrade_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Обратная сторона: при доступной схеме предупреждения быть не должно.

    Иначе оно превратится в постоянный шум, который перестанут читать.
    """
    from backend.core.config import get_settings

    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    try:
        registry = DatasetRegistry(root=tmp_path / "datasets")
        snapshot = registry.import_package(GOLDEN)

        assert not any("не проверены по схеме" in warning for warning in snapshot.warnings)
    finally:
        get_settings.cache_clear()
