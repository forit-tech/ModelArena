"""Применение сохранённой модели к новым данным.

Единственное правило, нарушение которого обесценивает всё остальное:
**на данных для предсказания ничего не обучается заново.** Ни импьютер, ни кодировщик,
ни масштабирование. Пересчитанная на новых строках медиана или пересобранный словарь
категорий дали бы другое числовое представление тех же значений — и предсказание
перестало бы соответствовать модели, оставшись при этом правдоподобным.

Поэтому здесь вызываются только `predict` и `predict_proba` уже обученного конвейера,
а подготовка входа повторяет ровно то, что делалось перед обучением: те же колонки,
в том же порядке, с тем же приведением категорий к строкам.

Схема проверяется **до** предсказания. Недостающая колонка — отказ: молча подставить
пропуск значило бы выдать ответ по данным, которых нет. Лишняя колонка — предупреждение:
она отбрасывается, но об этом сказано.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from backend.artifacts.manifest import ArtifactError, FeatureSchema


@dataclass
class SchemaCheck:
    """Что не так со входными данными относительно того, чего ждёт модель."""

    ok: bool
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "missing": self.missing,
            "unexpected": self.unexpected,
            "notes": self.notes,
        }


def check_schema(columns: list[str], schema: FeatureSchema) -> SchemaCheck:
    present = set(columns)
    missing = [name for name in schema.columns if name not in present]
    unexpected = [name for name in columns if name not in set(schema.columns)]
    notes: list[str] = []

    if unexpected:
        notes.append(
            "Лишние колонки не используются моделью и отброшены: "
            + ", ".join(unexpected[:10])
            + ("…" if len(unexpected) > 10 else "")
            + ". Модель обучена на фиксированном наборе признаков, и добавить новый "
            "без переобучения нельзя."
        )

    if missing:
        notes.append(
            "Без этих колонок предсказание невозможно: подставить вместо них пропуск "
            "значило бы ответить по данным, которых нет."
        )

    return SchemaCheck(ok=not missing, missing=missing, unexpected=unexpected, notes=notes)


def prepare_frame(frame: pl.DataFrame, schema: FeatureSchema) -> tuple[Any, SchemaCheck]:
    """Привести вход к тому виду, который конвейер видел при обучении.

    Порядок колонок восстанавливается по манифесту, а категории приводятся к строкам
    тем же способом, что и перед обучением. Без этого `True` из нового файла и `"True"`
    из обучающего оказались бы разными категориями, и кодировщик посчитал бы значение
    незнакомым.
    """
    from backend.preprocessing.profiles import normalize_categoricals

    check = check_schema(list(frame.columns), schema)

    if not check.ok:
        raise ArtifactError(
            "schema_mismatch",
            "В данных нет колонок, которых ждёт модель: " + ", ".join(check.missing),
        )

    prepared = frame.select(schema.columns).to_pandas()
    return normalize_categoricals(prepared, schema.categorical), check


def predict_frame(
    pipeline: Any,
    manifest: dict[str, Any],
    frame: pl.DataFrame,
) -> dict[str, Any]:
    """Предсказать на новых данных обученной моделью.

    Ни одного вызова `fit`: конвейер только преобразует и предсказывает.
    """
    schema = FeatureSchema.from_dict(manifest.get("feature_schema") or {})
    task = dict(manifest.get("task") or {})
    task_type = str(task.get("task_type", ""))
    class_labels = [str(label) for label in (task.get("class_labels") or [])]
    prepared, check = prepare_frame(frame, schema)

    try:
        predicted = np.asarray(pipeline.predict(prepared))
    except Exception as error:
        raise ArtifactError(
            "prediction_failed",
            f"Модель не смогла предсказать на этих данных: {type(error).__name__}. "
            "Чаще всего причина — значения, несовместимые с типом колонки при обучении.",
        ) from error

    rows: list[dict[str, Any]] = []
    probabilities = _probabilities(pipeline, prepared, class_labels, task_type)

    for position in range(len(predicted)):
        entry: dict[str, Any] = {
            "index": position,
            "prediction": (
                float(predicted[position])
                if task_type == "regression"
                else str(predicted[position])
            ),
        }

        if probabilities is not None:
            entry["probabilities"] = {
                label: round(float(probabilities[position, column]), 6)
                for column, label in enumerate(class_labels)
            }

        rows.append(entry)

    return {
        "task_type": task_type,
        "n_rows": len(rows),
        "schema": check.to_dict(),
        "class_labels": class_labels if task_type != "regression" else [],
        "has_probabilities": probabilities is not None,
        "predictions": rows,
    }


def _probabilities(
    pipeline: Any, prepared: Any, class_labels: list[str], task_type: str
) -> np.ndarray | None:
    """Вероятности в порядке меток из манифеста, а не в порядке `classes_` модели.

    Порядок столбцов — то место, где ошибка даёт правдоподобный неверный ответ:
    вероятность одного класса окажется подписана именем другого, и заметить это
    по величине невозможно.
    """
    if task_type == "regression" or not class_labels:
        return None

    try:
        raw = np.asarray(pipeline.predict_proba(prepared))
    except (AttributeError, NotImplementedError):
        #модель вероятностей не даёт — это «неприменимо», а не сбой
        return None

    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    classes = [str(value) for value in getattr(model, "classes_", [])]

    if not classes or raw.ndim != 2:
        return None

    from backend.arena.predictions import PredictionError, align_probabilities

    try:
        aligned, _ = align_probabilities(raw, classes, class_labels)
    except PredictionError as error:
        #несоответствие классов модели и задачи — это отказ, а не повод вернуть None:
        #молча отбросив вероятности, мы бы скрыли, что артефакт не соответствует задаче
        raise ArtifactError(
            "probability_mismatch",
            f"Вероятности не сопоставляются с классами задачи: {error.message}",
        ) from error

    return aligned


def predictions_to_frame(result: dict[str, Any]) -> pl.DataFrame:
    """Собрать таблицу предсказаний для выгрузки."""
    columns: dict[str, Any] = {
        "index": [item["index"] for item in result["predictions"]],
        "prediction": [item["prediction"] for item in result["predictions"]],
    }

    if result.get("has_probabilities"):
        for label in result["class_labels"]:
            columns[f"proba__{label}"] = [
                item["probabilities"][label] for item in result["predictions"]
            ]

    return pl.DataFrame(columns)
