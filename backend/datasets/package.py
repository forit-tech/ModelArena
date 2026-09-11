"""Чтение Dataset Package `dataarena.package/1`.

Реализация написана **по тексту спецификации** `contracts/dataset-package/v1/dataset-package-v1.md`
и не импортирует эталонные `reference/*.py`. Это не формальность: именно независимая
реализация по документу нашла три дефекта контракта, которые проверка эталона самим собой
найти не могла в принципе. Провенанс сохраняется — при обновлении контракта модуль правится
по документу, а не копированием из эталона.

Владелец контракта — DataArena (D-11). Здесь только сторона потребителя:
что мы обязаны проверить и на что обязаны отказать.

Ключевые правила, которые легко потерять при рефакторинге:

* вход отпечатка — **только** таблица из `data/*.parquet`; `schema.json` входом не является (§D);
* при этом `schema.json`, если он есть, обязан описывать те же данные — иначе пакет невалиден;
* `sha256` файла и отпечаток датасета отвечают на **разные** вопросы (D-14) и не смешиваются;
* незнакомый алгоритм отпечатка — отказ, а не чтение с предупреждением: идентичность,
  которую нельзя проверить, не должна выглядеть проверенной.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from jsonschema import Draft202012Validator

from backend.core.errors import PackageError

SUPPORTED_FORMAT = "dataarena.package"
SUPPORTED_MAJOR = 1
FINGERPRINT_ALGORITHM = "dataarena-logical-sha256-v1"

#лимиты §H защищают разбор метаданных: JSON читается в память целиком до любых проверок
LIMITS = {
    "manifest_bytes": 1 * 1024 * 1024,
    "metadata_part_bytes": 16 * 1024 * 1024,
    "metadata_total_bytes": 64 * 1024 * 1024,
    "file_count": 10_000,
    "decompression_ratio": 100,
}
METADATA_PARTS = {
    "schema.json": "schema.schema.json",
    "profile.json": "profile.schema.json",
    "lineage.json": "lineage.schema.json",
    "recipe.json": "recipe.schema.json",
}
_INTEGER_TYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_CANONICAL_QUIET_NAN = 0x7FF8000000000000


@dataclass(frozen=True)
class LoadedPackage:
    #результат чтения: данные плюс всё, что о них заявлено. Заявленное отделено от проверенного
    frame: pl.DataFrame
    manifest: dict[str, Any]
    schema: dict[str, Any] | None
    profile: dict[str, Any] | None
    lineage: dict[str, Any] | None
    recipe: dict[str, Any] | None
    fingerprint: str
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or "dataset")

    @property
    def row_key(self) -> list[str]:
        #устойчивая ссылка на строку между сервисами, если producer её объявил.
        #Признаком она не становится — это идентификатор, а не сигнал
        return list(self.manifest.get("data", {}).get("row_key") or [])


# --------------------------------------------------------------------------- отпечаток


def logical_type(dtype: pl.DataType) -> str:
    #нормативная таблица §D. Типа, которого в ней нет, быть не должно: producer обязан привести
    #колонку к поддерживаемому типу, а не отдать её нам «как получилось»
    if dtype == pl.Boolean:
        return "boolean"
    if dtype in _INTEGER_TYPES:
        return "integer"
    if dtype in (pl.Float32, pl.Float64):
        return "float"
    if isinstance(dtype, pl.Decimal):
        return "decimal"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "datetime"
    if isinstance(dtype, pl.Duration):
        return "duration"
    if dtype == pl.Time:
        return "time"
    if dtype == pl.String:
        return "string"
    if dtype == pl.Binary:
        return "binary"

    raise PackageError("package_unsupported_dtype", f"Тип колонки {dtype} не описан контрактом.")


def _nfc(text: str) -> str:
    #«café» одним кодовым пунктом и «cafe» + комбинирующий акцент — один и тот же текст
    return unicodedata.normalize("NFC", text)


def _length_prefixed(items: list[bytes]) -> bytes:
    #блок длин перед полезной нагрузкой: без него ["ab","c"] и ["a","bc"] дали бы один хеш
    lengths = np.array([len(item) for item in items], dtype="<u4")
    return lengths.tobytes() + b"".join(items)


def _null_mask_bytes(series: pl.Series) -> bytes:
    mask = series.is_null().to_numpy().astype(bool)
    return np.packbits(mask, bitorder="little").tobytes()


def _value_bytes(series: pl.Series, logical: str) -> bytes:
    #пропуски заменяются нейтральным значением: сам факт пропуска несёт отдельно хешируемая маска
    if logical == "integer":
        return series.fill_null(0).cast(pl.Int64).to_numpy().astype("<i8").tobytes()

    if logical == "float":
        values = series.cast(pl.Float64).to_numpy().astype("<f8").copy()
        nan_positions = np.isnan(values)
        values[nan_positions] = 0.0
        values[values == 0.0] = 0.0                     # снимает знак у -0.0: по IEEE-754 они равны
        raw = values.view("<u8").copy()
        raw[nan_positions] = _CANONICAL_QUIET_NAN
        raw[series.is_null().to_numpy().astype(bool)] = 0
        return raw.tobytes()

    if logical == "decimal":
        #масштаб нормализуется: 1.50 и 1.5 — одно число. format(..., "f") убирает экспоненту,
        #в которую normalize() уводит круглые числа: Decimal("100").normalize() -> 1E+2
        items = [
            b"" if value is None else format(Decimal(str(value)).normalize(), "f").encode("ascii")
            for value in series.to_list()
        ]
        return _length_prefixed(items)

    if logical == "boolean":
        return series.fill_null(False).to_numpy().astype(np.uint8).tobytes()

    if logical == "date":
        return series.cast(pl.Int32).fill_null(0).to_numpy().astype("<i4").tobytes()

    if logical == "datetime":
        casted = series

        if getattr(series.dtype, "time_zone", None):
            casted = series.dt.convert_time_zone("UTC").dt.replace_time_zone(None)

        casted = casted.cast(pl.Datetime("us")).cast(pl.Int64).fill_null(0)
        return casted.to_numpy().astype("<i8").tobytes()

    if logical == "duration":
        return series.cast(pl.Duration("us")).cast(pl.Int64).fill_null(0).to_numpy().astype("<i8").tobytes()

    if logical == "time":
        return series.cast(pl.Int64).fill_null(0).to_numpy().astype("<i8").tobytes()

    if logical == "string":
        return _length_prefixed(
            [b"" if value is None else _nfc(value).encode("utf-8") for value in series.to_list()]
        )

    if logical == "binary":
        return _length_prefixed([b"" if value is None else bytes(value) for value in series.to_list()])

    raise PackageError("package_unsupported_dtype", f"Кодирование для {logical} не описано.")


def content_fingerprint(frame: pl.DataFrame) -> str:
    #отпечаток логического содержимого: не зависит от версии библиотеки, compression,
    #dictionary encoding и раскладки Parquet (D-14)
    digest = hashlib.sha256()
    digest.update(FINGERPRINT_ALGORITHM.encode("ascii") + b"\x0A")
    digest.update(str(frame.height).encode("ascii") + b"\x1E")
    digest.update(str(frame.width).encode("ascii") + b"\x1D")

    for name, dtype in frame.schema.items():
        logical = logical_type(dtype)
        name_utf8 = _nfc(name).encode("utf-8")          # NFC применяется и к именам колонок
        digest.update(struct.pack("<I", len(name_utf8)) + name_utf8)
        digest.update(logical.encode("ascii") + b"\x1E")
        digest.update(hashlib.sha256(_null_mask_bytes(frame[name])).digest())
        digest.update(hashlib.sha256(_value_bytes(frame[name], logical)).digest())
        digest.update(b"\x1D")

    return digest.hexdigest()


# --------------------------------------------------------------------------- чтение пакета


def safe_member(package: Path, relative: str) -> Path:
    """Собрать путь внутри пакета из **недоверенной** строки манифеста.

    Манифест пишет другой сервис, а пакет может прийти файлом откуда угодно. Без этой
    проверки `parts[].path` вида `../../secret` заставлял бы ModelArena читать файлы
    за пределами пакета и подтверждать их содержимое сверкой `sha256` — то есть работать
    оракулом по чужой файловой системе.

    Проверяется **итоговый разрешённый путь**, а не исходная строка: `a/../../b`
    выглядит безобидно посимвольно и выходит наружу после нормализации.
    """
    if not relative or "\x00" in relative:
        raise PackageError("package_unsafe_path", "Пустое или некорректное имя файла в манифесте.")

    candidate = Path(relative.replace("\\", "/"))

    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise PackageError("package_unsafe_path", f"Абсолютный путь в манифесте: {relative}")

    root = package.resolve()
    resolved = (root / candidate).resolve()

    if resolved != root and root not in resolved.parents:
        raise PackageError("package_unsafe_path", f"Путь выходит за пределы пакета: {relative}")

    return resolved


def _extract_archive(archive: Path, workspace: Path, data_budget_bytes: int) -> Path:
    #архив проверяется ДО распаковки: иначе защита от zip-бомбы срабатывает после того,
    #как диск уже занят
    total_uncompressed = 0
    total_compressed = 0

    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()

        if len(entries) > LIMITS["file_count"]:
            raise PackageError("package_limit_exceeded", "В архиве слишком много файлов.")

        for entry in entries:
            name = entry.filename.replace("\\", "/")

            if name.startswith("/") or ".." in Path(name).parts or (len(name) > 1 and name[1] == ":"):
                raise PackageError("package_unsafe_path", f"Небезопасный путь в архиве: {entry.filename}")

            if name.lower().endswith((".zip", ".dapkg.zip")):
                raise PackageError("package_unsafe_path", "Вложенные архивы запрещены.")

            total_uncompressed += entry.file_size
            total_compressed += entry.compress_size

        #абсолютный предел проверяется ОТДЕЛЬНО от коэффициента: архив, записанный без сжатия,
        #даёт коэффициент 1:1 при любом объёме и проходил бы проверку соотношения насквозь
        if total_uncompressed > data_budget_bytes:
            raise PackageError(
                "package_limit_exceeded",
                f"Распакованный размер {total_uncompressed / 1024 / 1024:.0f} МБ превышает "
                f"предел потребителя {data_budget_bytes / 1024 / 1024:.0f} МБ. "
                "Это ограничение ModelArena, а не дефект пакета.",
            )

        #нулевой compress_size у всех записей означал бы деление на ноль; трактуем как 1:1
        ratio = total_uncompressed / total_compressed if total_compressed else 1.0

        if ratio > LIMITS["decompression_ratio"]:
            raise PackageError(
                "package_limit_exceeded",
                f"Коэффициент распаковки {ratio:.0f}:1 превышает предел "
                f"{LIMITS['decompression_ratio']}:1.",
            )

        source.extractall(workspace)

    if (workspace / "manifest.json").exists():
        return workspace

    roots = [child for child in workspace.iterdir() if child.is_dir()]
    return roots[0] if len(roots) == 1 else workspace


def _read_manifest(package: Path) -> dict[str, Any]:
    manifest_path = package / "manifest.json"

    if not manifest_path.exists():
        raise PackageError("package_part_missing", "В пакете нет manifest.json.")

    if manifest_path.stat().st_size > LIMITS["manifest_bytes"]:
        raise PackageError("package_limit_exceeded", "manifest.json превышает 1 MiB.")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PackageError("package_part_invalid", f"manifest.json не разбирается: {error}") from error

    if not isinstance(manifest, dict):
        raise PackageError("package_part_invalid", "manifest.json должен быть объектом.")

    if manifest.get("format") != SUPPORTED_FORMAT:
        raise PackageError(
            "package_format_unsupported",
            f"Ожидался формат {SUPPORTED_FORMAT}, получен {manifest.get('format')!r}.",
        )

    #отсутствие обязательного ключа — это дефект пакета, а не сбой сервиса: без явной
    #проверки KeyError уходил бы наружу пятисоткой без объяснения
    for key in ("format_version", "data"):
        if key not in manifest:
            raise PackageError("package_part_invalid", f"В manifest.json нет обязательного поля {key}.")

    data = manifest["data"]

    if not isinstance(data, dict) or not isinstance(data.get("files"), list) or not data["files"]:
        raise PackageError(
            "package_part_invalid", "manifest.data.files должен быть непустым списком файлов."
        )

    return manifest


def _check_version(manifest: dict[str, Any], warnings: list[str]) -> None:
    raw = str(manifest["format_version"])

    try:
        major, minor = (int(part) for part in raw.split(".")[:2])
    except ValueError as error:
        raise PackageError(
            "package_part_invalid", f"format_version «{raw}» не является semver-версией."
        ) from error

    if major != SUPPORTED_MAJOR:
        raise PackageError(
            "package_version_unsupported",
            f"ModelArena понимает {SUPPORTED_FORMAT} версии {SUPPORTED_MAJOR}.x, "
            f"в пакете {manifest['format_version']}.",
        )

    if minor > 0:
        warnings.append(
            f"Пакет собран более новой minor-версией {manifest['format_version']}: "
            "незнакомые поля игнорируются."
        )


def _check_integrity(package: Path, manifest: dict[str, Any]) -> None:
    declared_paths = [part["path"] for part in manifest.get("parts", [])]

    if len(declared_paths) != len(set(declared_paths)):
        raise PackageError("package_duplicate_entry", "Один путь объявлен в parts дважды.")

    for part in manifest.get("parts", []):
        if not isinstance(part, dict) or "path" not in part or "sha256" not in part:
            raise PackageError("package_part_invalid", "Запись в parts без path или sha256.")

        path = safe_member(package, str(part["path"]))

        if not path.exists():
            raise PackageError("package_part_missing", f"Объявленный файл отсутствует: {part['path']}")

        if hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
            raise PackageError(
                "package_integrity_failed", f"sha256 не совпал для {part['path']}: файл изменён."
            )

    for name in manifest["data"]["files"]:
        if not safe_member(package, str(name)).exists():
            raise PackageError("package_part_missing", f"Файл данных отсутствует: {name}")


def _read_metadata_parts(
    package: Path,
    schema_directory: Path | None,
    warnings: list[str],
) -> dict[str, dict[str, Any] | None]:
    parts: dict[str, dict[str, Any] | None] = dict.fromkeys(METADATA_PARTS)
    total = 0

    for name, schema_name in METADATA_PARTS.items():
        path = package / name

        if not path.exists():
            continue

        size = path.stat().st_size
        total += size

        if size > LIMITS["metadata_part_bytes"] or total > LIMITS["metadata_total_bytes"]:
            raise PackageError("package_limit_exceeded", f"{name} превышает лимит метаданных.")

        payload = json.loads(path.read_text(encoding="utf-8"))

        if schema_directory is not None:
            validator = Draft202012Validator(
                json.loads((schema_directory / schema_name).read_text(encoding="utf-8"))
            )
            error = next(validator.iter_errors(payload), None)

            if error is not None:
                #часть, не прошедшая свою JSON Schema, дальше не разбирается: иначе на один
                #дефект приходит каскад кодов и непонятно, на какой реагировать
                raise PackageError("package_part_invalid", f"{name}: {error.message}")

        parts[name] = payload

    profile = parts["profile.json"]

    if profile and (profile.get("computed_on") or {}).get("sampled"):
        warnings.append(
            "Профиль в пакете посчитан по выборке: статистики приблизительны "
            "и не должны выдаваться за точные."
        )

    return parts


def _read_frame(package: Path, manifest: dict[str, Any]) -> pl.DataFrame:
    #порядок строк нормативен: части конкатенируются в порядке manifest.data.files,
    #внутри части — порядок хранения. Без него не воспроизводится отпечаток
    frames = [pl.read_parquet(safe_member(package, str(name))) for name in manifest["data"]["files"]]

    if any(frame.schema != frames[0].schema for frame in frames):
        raise PackageError("package_parts_schema_mismatch", "Схемы частей данных различаются.")

    frame = pl.concat(frames, how="vertical")
    data = manifest["data"]

    if data.get("row_count_exact") and frame.height != data["row_count"]:
        raise PackageError(
            "package_row_count_mismatch",
            f"Заявлено строк {data['row_count']}, фактически {frame.height}.",
        )

    return frame


def _check_schema_consistency(schema: dict[str, Any] | None, frame: pl.DataFrame) -> None:
    #валидный пакет не может содержать две расходящиеся схемы. Вместе с правилом «вход отпечатка
    #только Parquet» это значит, что источники физически не могут разойтись
    if schema is None:
        return

    columns = schema["columns"]
    physical = [(name, logical_type(dtype)) for name, dtype in frame.schema.items()]
    declared = [(column["name"], column["logical_type"]) for column in columns]
    positions = [column.get("position") for column in columns]

    if any(
        position is not None and position != index
        for index, position in enumerate(positions[: len(physical)])
    ):
        raise PackageError("package_schema_mismatch", "position в schema.json не совпадает с порядком колонок.")

    if declared != physical:
        raise PackageError(
            "package_schema_mismatch",
            f"schema.json описывает колонки {[name for name, _ in declared]}, "
            f"а в parquet {[name for name, _ in physical]}.",
        )


def read_package(
    package_path: Path,
    schema_directory: Path | None = None,
    *,
    data_budget_bytes: int | None = None,
) -> LoadedPackage:
    """Прочитать Dataset Package и вернуть данные вместе с проверенными метаданными.

    `schema_directory` — каталог JSON Schema контракта. Если он не передан, проверка
    метаданных по схеме пропускается: остальные проверки §H выполняются в полном объёме.

    `data_budget_bytes` — операционный предел потребителя на распакованный объём данных.
    Контракт его не задаёт намеренно: датасет законно бывает огромным (D-17, §9.2).
    """
    from backend.core.config import get_settings

    budget = data_budget_bytes or get_settings().max_package_data_bytes
    workspace: Path | None = None

    try:
        target = package_path

        if package_path.suffix == ".zip":
            workspace = Path(tempfile.mkdtemp(prefix="dapkg_"))
            target = _extract_archive(package_path, workspace, budget)

        warnings: list[str] = []
        manifest = _read_manifest(target)
        _check_version(manifest, warnings)
        _check_integrity(target, manifest)
        parts = _read_metadata_parts(target, schema_directory, warnings)
        frame = _read_frame(target, manifest)
        _check_schema_consistency(parts["schema.json"], frame)

        declared = manifest["data"].get("content_fingerprint") or {}

        if declared.get("algorithm") != FINGERPRINT_ALGORITHM:
            raise PackageError(
                "package_fingerprint_algorithm_unknown",
                f"Алгоритм отпечатка {declared.get('algorithm')!r} не поддерживается. "
                "Принять пакет без проверки идентичности можно только явным решением "
                "пользователя, и такой снимок будет помечен как непроверенный.",
            )

        fingerprint = content_fingerprint(frame)

        if fingerprint != declared.get("value"):
            raise PackageError(
                "package_fingerprint_mismatch",
                "Отпечаток содержимого не совпал с манифестом: пакет описывает не те данные, "
                "которые в нём лежат.",
            )

        return LoadedPackage(
            frame=frame,
            manifest=manifest,
            schema=parts["schema.json"],
            profile=parts["profile.json"],
            lineage=parts["lineage.json"],
            recipe=parts["recipe.json"],
            fingerprint=fingerprint,
            warnings=warnings,
        )
    finally:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
