"""Сохранённые модели: карточка артефакта, проверка целостности, применение к новым данным.

Граница доверия проходит здесь. `joblib` при загрузке **исполняет код**, поэтому загружается
только артефакт, созданный этим же приложением и лежащий внутри его каталога прогонов.
Загрузки произвольного файла с диска или из сети этот слой не предоставляет — и не должен:
обещание «безопасно загрузим любую модель» форматом не обеспечено.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from backend.arena.runner import get_runner
from backend.artifacts.inference import predict_frame, predictions_to_frame
from backend.artifacts.manifest import (
    ArtifactError,
    environment_warnings,
    load_artifact,
    verify,
)
from backend.core.errors import AppError, NotFoundError, ValidationError
from backend.datasets.io import read_table
from backend.diagnostics.importance import DEFAULT_REPEATS

router = APIRouter(prefix="/models", tags=["models"])

#предел на файл для пакетного предсказания: он читается в память целиком
MAX_BATCH_BYTES = 64 * 1024 * 1024


class SinglePrediction(BaseModel):
    #значения признаков: ключ — имя колонки, как в схеме артефакта
    values: dict[str, Any] = Field(default_factory=dict)


def _artifact_directory(run_id: str, contender_key: str) -> Path:
    runner = get_runner()
    record = runner.get(run_id)
    known = {item.contender_key for item in record.contenders}

    if contender_key not in known:
        raise NotFoundError(f"В прогоне {run_id} нет участника «{contender_key}».")

    directory = runner.store.contender_path(run_id, contender_key)

    if not (directory / "model.json").exists():
        raise NotFoundError(
            f"У участника «{contender_key}» нет сохранённой модели. Артефакт создаётся "
            "только для успешно завершившихся участников."
        )

    return directory


def _as_app_error(error: ArtifactError) -> AppError:
    """Перевести отказ артефакта в ответ API с сохранением машинного кода."""
    failure = ValidationError(error.message)
    failure.code = error.code
    failure.status_code = 404 if error.code == "artifact_missing" else 400
    return failure


@router.get(
    "/{run_id}/{contender_key}",
    summary="Карточка сохранённой модели",
    description=(
        "Манифест артефакта: чему модель обучена, каких колонок ждёт, что означает её выход, "
        "какими версиями библиотек записана. Целостность сверяется по контрольной сумме "
        "до какой-либо загрузки кода."
    ),
)
def get_model(run_id: str, contender_key: str) -> dict[str, Any]:
    directory = _artifact_directory(run_id, contender_key)

    try:
        manifest = verify(directory)
    except ArtifactError as error:
        raise _as_app_error(error) from error

    return {
        "run_id": run_id,
        "contender_key": contender_key,
        "manifest": manifest,
        #расхождение версий не мешает работать, но молчать о нём нельзя
        "environment_warnings": environment_warnings(manifest),
        "trust_boundary": (
            "Артефакт загружается через joblib, который исполняет код при десериализации. "
            "Поэтому загружаются только артефакты, созданные этим приложением, и только "
            "после сверки контрольной суммы. Загрузка чужих файлов не поддерживается."
        ),
    }


@router.get(
    "/{run_id}/{contender_key}/importance",
    summary="Влияние признаков",
    description=(
        "Перестановочная важность на holdout — части, которую сохранённая модель не видела. "
        "Модель не переобучается: перестановка меняет вход, а не модель. Это измерение "
        "опоры модели, а не утверждение о причинной связи."
    ),
)
def get_importance(
    run_id: str,
    contender_key: str,
    metric: str | None = None,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
) -> dict[str, Any]:
    from backend.arena.metrics import read_predictions
    from backend.datasets.registry import DatasetRegistry
    from backend.diagnostics.importance import permutation_importance_for

    runner = get_runner()
    record = runner.get(run_id)
    directory = _artifact_directory(run_id, contender_key)
    path = runner.store.predictions_path(run_id, contender_key)

    if not path.exists():
        raise NotFoundError(f"У участника «{contender_key}» нет сохранённых предсказаний.")

    try:
        pipeline, manifest = load_artifact(directory)
    except ArtifactError as error:
        raise _as_app_error(error) from error

    registry = DatasetRegistry()
    snapshot = registry.load_frame(str(record.spec.get("dataset_id", "")), with_row_id=True)

    return {
        "run_id": run_id,
        "contender_key": contender_key,
        "importance": permutation_importance_for(
            pipeline,
            manifest,
            snapshot,
            read_predictions(path),
            metric=metric,
            repeats=repeats,
            seed=seed,
        ),
    }


@router.post(
    "/{run_id}/{contender_key}/predict",
    summary="Предсказание по одной строке",
    description=(
        "Применяет сохранённую модель к одному набору значений. Препроцессинг берётся "
        "из артефакта и заново не обучается."
    ),
)
def predict_one(run_id: str, contender_key: str, request: SinglePrediction) -> dict[str, Any]:
    directory = _artifact_directory(run_id, contender_key)

    if not request.values:
        raise ValidationError("Не передано ни одного значения признака.")

    try:
        pipeline, manifest = load_artifact(directory)
        frame = pl.DataFrame([request.values])
        result = predict_frame(pipeline, manifest, frame)
    except ArtifactError as error:
        raise _as_app_error(error) from error

    return {"run_id": run_id, "contender_key": contender_key, "result": result}


@router.post(
    "/{run_id}/{contender_key}/predict-batch",
    summary="Предсказание по файлу",
    description=(
        "Применяет сохранённую модель к загруженной таблице. Схема проверяется до "
        "предсказания: недостающая колонка — отказ, лишняя — предупреждение."
    ),
)
async def predict_batch(
    run_id: str, contender_key: str, file: UploadFile = File(...)
) -> dict[str, Any]:
    directory = _artifact_directory(run_id, contender_key)
    payload = await file.read(MAX_BATCH_BYTES + 1)

    if len(payload) > MAX_BATCH_BYTES:
        raise ValidationError(
            f"Файл больше {MAX_BATCH_BYTES // (1024 * 1024)} МБ. Для пакетного предсказания "
            "он читается в память целиком, поэтому предел жёсткий."
        )

    frame = _read_upload(payload, file.filename or "data.csv")

    try:
        pipeline, manifest = load_artifact(directory)
        result = predict_frame(pipeline, manifest, frame)
    except ArtifactError as error:
        raise _as_app_error(error) from error

    #в ответ уходит и таблица целиком: её же пользователь выгружает файлом
    table = predictions_to_frame(result)
    buffer = io.BytesIO()
    table.write_csv(buffer)

    return {
        "run_id": run_id,
        "contender_key": contender_key,
        "result": result,
        "csv": buffer.getvalue().decode("utf-8"),
    }


def _read_upload(payload: bytes, filename: str) -> pl.DataFrame:
    import tempfile

    suffix = Path(filename).suffix.lower() or ".csv"

    if suffix not in {".csv", ".tsv", ".parquet", ".json", ".xlsx"}:
        raise ValidationError(
            f"Формат «{suffix}» не поддерживается для пакетного предсказания. "
            "Ожидаются csv, tsv, parquet, json или xlsx."
        )

    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / f"batch{suffix}"
        path.write_bytes(payload)
        return read_table(path)
