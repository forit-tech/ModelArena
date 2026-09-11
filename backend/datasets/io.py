"""Чтение табличных файлов — единственная точка входа сырых данных.

Нужна для standalone-режима (D-13): ModelArena принимает файл напрямую, но не редактирует
его. Любой новый формат добавляется здесь, и все слои получают его автоматически.

Логика определения разделителя и кодировки перенесена из AutoDataAnalysis без изменений:
она проверена на реальных выгрузках, где встречаются и UTF-16, и Windows-1251, и точка
с запятой вместо запятой.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import polars as pl

from backend.core.errors import DatasetReadError, UnsupportedFormatError

SUPPORTED_EXTENSIONS = frozenset({".csv", ".parquet", ".json", ".jsonl", ".ndjson", ".xlsx"})
#перебор кодировок и разделителей стоит дорого на больших файлах, поэтому схема выводится
#по ограниченной выборке строк
SCHEMA_INFERENCE_ROWS = 10_000


def supported_formats_text() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def ensure_supported_extension(file_name: str) -> str:
    #расширение не гарантирует содержимого, но дёшево отсекает очевидно посторонние загрузки
    suffix = Path(file_name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Неподдерживаемый формат файла. Поддерживаются: {supported_formats_text()}.",
            {"file_name": file_name, "supported": sorted(SUPPORTED_EXTENSIONS)},
        )

    return suffix


def read_table(file_path: Path) -> pl.DataFrame:
    resolved = file_path.resolve()
    suffix = ensure_supported_extension(resolved.name)

    try:
        if suffix == ".csv":
            return _read_csv(resolved)

        if suffix == ".parquet":
            return pl.read_parquet(resolved)

        if suffix in {".jsonl", ".ndjson"}:
            return pl.read_ndjson(resolved)

        if suffix == ".json":
            return _read_json(resolved)

        return _read_excel(resolved)
    except DatasetReadError:
        raise
    except Exception as error:
        raise DatasetReadError(_describe_failure(resolved.name, error)) from error


def _describe_failure(file_name: str, error: Exception) -> str:
    #многострочные подсказки парсеров выглядят в интерфейсе как утечка внутренностей
    text = str(error).strip()
    first_line = text.splitlines()[0] if text else error.__class__.__name__
    return f"Не удалось прочитать датасет «{file_name}»: {first_line[:200]}"


def _read_excel(file_path: Path) -> pl.DataFrame:
    #новые версии Polars могут вернуть Series или словарь листов — нормализуем явно
    result = pl.read_excel(file_path)

    if isinstance(result, pl.DataFrame):
        return result

    if isinstance(result, pl.Series):
        return result.to_frame()

    if isinstance(result, dict) and result:
        first = next(iter(result.values()))
        return first if isinstance(first, pl.DataFrame) else first.to_frame()

    raise DatasetReadError("Не удалось прочитать Excel-файл: лист не содержит таблицы.")


def _read_json(file_path: Path) -> pl.DataFrame:
    #обычный JSON-массив объектов или newline-delimited: выбор строится на неудачной попытке
    try:
        return pl.read_json(file_path)
    except Exception:  # noqa: BLE001 - см. комментарий выше
        return pl.read_ndjson(file_path)


def _read_csv(file_path: Path) -> pl.DataFrame:
    detected = _detect_separator(file_path)
    separators = list(dict.fromkeys([detected, ",", ";", "\t", "|"]))
    last_error: Exception | None = None

    #быстрый путь без загрузки файла целиком — подходит корректному UTF-8
    for separator in separators:
        try:
            return pl.read_csv(
                file_path,
                separator=separator,
                infer_schema_length=SCHEMA_INFERENCE_ROWS,
                ignore_errors=True,
                truncate_ragged_lines=True,
                try_parse_dates=True,
            )
        except Exception as error:  # noqa: BLE001 - перебор построен на неудачных попытках
            last_error = error

    #не UTF-8: один раз декодируем распространённой legacy-кодировкой и повторяем подбор
    decoded = _decode_csv(file_path)

    for separator in separators:
        try:
            return pl.read_csv(
                io.StringIO(decoded),
                separator=separator,
                infer_schema_length=SCHEMA_INFERENCE_ROWS,
                ignore_errors=True,
                truncate_ragged_lines=True,
                try_parse_dates=True,
            )
        except Exception as error:  # noqa: BLE001 - см. выше
            last_error = error

    raise DatasetReadError(_describe_failure(file_path.name, last_error or ValueError("unknown")))


def _decode_csv(file_path: Path) -> str:
    #выбор между cp1251 и cp1252 делается по доле кириллицы: иначе русский текст
    #превращается в нечитаемые символы, а файл при этом «успешно» читается
    content = file_path.read_bytes()

    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")

    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass

    candidates: dict[str, str] = {}

    for encoding in ("cp1251", "cp1252", "latin-1"):
        try:
            candidates[encoding] = content.decode(encoding)
        except UnicodeDecodeError:
            continue

    cp1251_text = candidates.get("cp1251")

    if cp1251_text:
        letters = sum(character.isalpha() for character in cp1251_text)
        cyrillic = sum("Ѐ" <= character <= "ӿ" for character in cp1251_text)

        if cyrillic >= 3 and letters > 0 and cyrillic / letters >= 0.05:
            return cp1251_text

    for encoding in ("cp1252", "latin-1", "cp1251"):
        if encoding in candidates:
            return candidates[encoding]

    raise DatasetReadError("Не удалось определить текстовую кодировку CSV-файла.")


def _detect_separator(file_path: Path) -> str:
    sample = file_path.read_bytes()[:8192]
    text = ""

    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            text = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if not text:
        return ","

    try:
        return csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","
