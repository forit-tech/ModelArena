"""Датасеты: импорт, список, карточка, предпросмотр строк."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, File, Query, UploadFile

from backend.core.config import get_settings
from backend.core.errors import ValidationError
from backend.datasets.io import SUPPORTED_EXTENSIONS, supported_formats_text
from backend.datasets.profile import build_profile
from backend.datasets.registry import DatasetRegistry, DatasetSnapshot

router = APIRouter(prefix="/datasets", tags=["datasets"])

PACKAGE_SUFFIXES = (".dapkg.zip", ".zip", ".dapkg")
MAX_PREVIEW_ROWS = 200


def _safe_upload_name(raw: str | None) -> str:
    """Имя загруженного файла — недоверенный ввод.

    Оно используется только для выбора обработчика и как подпись снимка, но всё равно
    попадает во временный путь. `Path(...).name` отсекает каталоги, однако оставляет
    вырожденные случаи: «..» и «.» дают пустую строку, и путь схлопывается в сам каталог.
    """
    raw_name = raw or ""

    if "\0" in raw_name:
        raise ValidationError("Имя файла содержит нулевой байт.")

    candidate = Path(raw_name.replace("\\", "/")).name.strip()

    if not candidate or candidate in {".", ".."}:
        return "dataset"

    #длинное имя упирается в предел файловой системы уже во временном каталоге
    return candidate[-180:]


def _registry() -> DatasetRegistry:
    #реестр создаётся на запрос: каталог артефактов берётся из настроек, а не из глобального
    #состояния, поэтому тесты могут работать в своём каталоге
    return DatasetRegistry()


@router.post(
    "",
    status_code=201,
    summary="Импортировать датасет",
    description=(
        "Принимает Dataset Package из DataArena (`.dapkg.zip`) или табличный файл напрямую. "
        "Пакет проверяется по контракту целиком: версия формата, целостность частей, "
        "согласованность схемы с данными и отпечаток содержимого."
    ),
)
async def import_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    file_name = _safe_upload_name(file.filename)
    settings = get_settings()
    workspace = Path(tempfile.mkdtemp(prefix="marena_upload_"))

    try:
        target = workspace / file_name
        size = 0

        with target.open("wb") as sink:
            #файл пишется потоково: датасет законно бывает большим, и читать его целиком
            #в память ради проверки размера бессмысленно
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > settings.max_upload_bytes:
                    raise ValidationError(
                        f"Файл больше допустимого размера "
                        f"({settings.max_upload_bytes // (1024 * 1024)} МБ)."
                    )

                sink.write(chunk)

        snapshot = _import_by_kind(target, file_name)
        return {"dataset": snapshot.to_dict()}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _import_by_kind(path: Path, file_name: str) -> DatasetSnapshot:
    registry = _registry()
    lowered = file_name.lower()

    if lowered.endswith(PACKAGE_SUFFIXES):
        return registry.import_package(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Неподдерживаемый формат «{file_name}». Ожидается Dataset Package (.dapkg.zip) "
            f"или таблица: {supported_formats_text()}."
        )

    return registry.import_file(path, name=Path(file_name).stem)


@router.get("", summary="Список импортированных датасетов")
def list_datasets() -> dict[str, Any]:
    snapshots = _registry().list_snapshots()
    return {"datasets": [snapshot.to_dict() for snapshot in snapshots], "total": len(snapshots)}


@router.get(
    "/{dataset_id}",
    summary="Карточка датасета",
    description="Снимок вместе с профилем колонок, посчитанным ModelArena по самим данным.",
)
def get_dataset(dataset_id: str) -> dict[str, Any]:
    registry = _registry()
    snapshot = registry.get(dataset_id)
    #профиль считаем сами, а не берём из пакета: чужая статистика могла устареть
    #относительно данных (D-9)
    profile = build_profile(registry.load_frame(dataset_id))

    return {"dataset": snapshot.to_dict(), "profile": profile.to_dict()}


@router.get(
    "/{dataset_id}/preview",
    summary="Страница строк",
    description="По сети уходит только видимая страница, а не весь датасет.",
)
def preview_dataset(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=MAX_PREVIEW_ROWS),
) -> dict[str, Any]:
    registry = _registry()
    snapshot = registry.get(dataset_id)
    frame = registry.load_frame(dataset_id)
    page = frame.slice(offset, limit)

    return {
        "dataset_id": dataset_id,
        "columns": frame.columns,
        "rows": _json_safe_rows(page),
        "offset": offset,
        "limit": limit,
        "total_rows": snapshot.row_count,
    }


@router.delete("/{dataset_id}", summary="Удалить снимок")
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    registry = _registry()
    registry.get(dataset_id)      # проверка существования до удаления каталога
    registry.delete(dataset_id)
    return {"deleted": dataset_id}


def _json_safe_rows(page: pl.DataFrame) -> list[dict[str, Any]]:
    #даты и десятичные значения не сериализуются в JSON сами по себе
    return [
        {name: _json_safe(value) for name, value in row.items()} for row in page.iter_rows(named=True)
    ]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return str(value)
