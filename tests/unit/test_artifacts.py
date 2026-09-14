"""Артефакт модели и применение к новым данным.

Самый опасный сценарий здесь не «не загрузилось», а «загрузилось и предсказало неверно».
Поэтому проверяются: отказ повреждённого и чужого артефакта до исполнения кода, полное
отсутствие дообучения на данных предсказания и сохранение порядка классов.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from backend.artifacts.inference import check_schema, predict_frame, predictions_to_frame
from backend.artifacts.manifest import (
    MANIFEST_FILE,
    MODEL_FILE,
    ArtifactError,
    FeatureSchema,
    build_manifest,
    environment_warnings,
    load_artifact,
    verify,
    write_artifact,
)
from backend.preprocessing.profiles import build_preprocessor, split_feature_types

LABELS = ["no", "yes"]


def _training_frame(rows: int = 80) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "amount": [float(index % 40) for index in range(rows)],
            "plan": [["basic", "pro"][index % 2] for index in range(rows)],
            "label": [LABELS[1] if index % 40 > 20 else LABELS[0] for index in range(rows)],
        }
    )


def _make_artifact(directory: Path, *, rows: int = 80) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    frame = _training_frame(rows)
    features = frame.select(["amount", "plan"]).to_pandas()
    types = split_feature_types(features)
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessor("scaled_onehot", types)),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(features, frame["label"].to_list())

    schema = FeatureSchema(
        columns=["amount", "plan"],
        dtypes={"amount": "float64", "plan": "object"},
        numeric=types.numeric,
        categorical=types.categorical,
        datetime=types.datetime,
    )
    directory.mkdir(parents=True, exist_ok=True)
    return write_artifact(
        directory,
        pipeline,
        build_manifest(
            run_id="run_0123456789abcdef0123",
            contender_key="logistic_regression",
            adapter_key="logistic_regression",
            label="Logistic Regression",
            params={"max_iter": 500},
            preprocessing_profile="scaled_onehot",
            task={
                "task_type": "binary",
                "target_column": "label",
                "class_labels": LABELS,
                "positive_label": "yes",
                "feature_columns": ["amount", "plan"],
            },
            dataset={"dataset_id": "ds_0123456789abcdef", "fingerprint": "fp"},
            protocol={"cv_splitter": "stratified_k_fold", "seed": 42},
            schema=schema,
            training_rows=rows,
            created_at="2026-09-14T00:00:00+00:00",
        ),
    )


# ---------------------------------------------------------------- целостность


def test_artifact_round_trips(tmp_path: Path) -> None:
    manifest = _make_artifact(tmp_path)

    assert manifest["model_sha256"]
    assert (tmp_path / MODEL_FILE).exists()

    pipeline, loaded = load_artifact(tmp_path)
    assert loaded["task"]["class_labels"] == LABELS
    #схема хранит порядок колонок: файл на входе может прийти с другим
    assert loaded["feature_schema"]["columns"] == ["amount", "plan"]
    assert hasattr(pipeline, "predict")


def test_corrupted_model_is_refused_before_loading_code(tmp_path: Path) -> None:
    """Проверка целостности идёт ДО загрузки: joblib исполняет код при десериализации."""
    _make_artifact(tmp_path)
    (tmp_path / MODEL_FILE).write_bytes("это не модель".encode())

    with pytest.raises(ArtifactError) as failure:
        load_artifact(tmp_path)

    assert failure.value.code == "artifact_corrupted"
    assert "сумма" in failure.value.message


def test_foreign_format_is_refused(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["format"] = "someone.elses.model"
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError) as failure:
        verify(tmp_path)

    assert failure.value.code == "artifact_foreign"


def test_unknown_major_version_is_refused_not_guessed(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["format_version"] = "2.0"
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError) as failure:
        verify(tmp_path)

    assert failure.value.code == "artifact_incompatible"


def test_missing_checksum_is_refused(tmp_path: Path) -> None:
    #без суммы проверить нечего, и «проверить нечем» не должно означать «всё в порядке»
    _make_artifact(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["model_sha256"] = ""
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError) as failure:
        verify(tmp_path)

    assert failure.value.code == "artifact_corrupted"


def test_version_mismatch_is_a_warning_not_a_refusal(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["environment"]["sklearn"] = "0.0.1"

    notes = environment_warnings(manifest)

    assert any("sklearn" in note for note in notes)
    #артефакт при этом остаётся читаемым: чаще всего он работает
    assert verify(tmp_path)


# ---------------------------------------------------------------- применение


def test_inference_never_refits_preprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ключевой регрессионный тест: на данных предсказания ничего не обучается.

    Пересчитанная на новых строках медиана или пересобранный словарь категорий дали бы
    другое числовое представление тех же значений — и предсказание перестало бы
    соответствовать модели, оставшись правдоподобным.
    """
    from sklearn.compose import ColumnTransformer

    _make_artifact(tmp_path)
    pipeline, manifest = load_artifact(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("на данных предсказания вызвано обучение")

    monkeypatch.setattr(ColumnTransformer, "fit", refuse)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", refuse)
    monkeypatch.setattr(Pipeline, "fit", refuse)

    result = predict_frame(
        pipeline, manifest, pl.DataFrame({"amount": [5.0, 35.0], "plan": ["pro", "basic"]})
    )

    assert result["n_rows"] == 2


def test_missing_column_is_refused_not_filled(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    pipeline, manifest = load_artifact(tmp_path)

    with pytest.raises(ArtifactError) as failure:
        predict_frame(pipeline, manifest, pl.DataFrame({"amount": [1.0]}))

    assert failure.value.code == "schema_mismatch"
    assert "plan" in failure.value.message


def test_extra_column_is_dropped_with_a_note(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    pipeline, manifest = load_artifact(tmp_path)

    result = predict_frame(
        pipeline,
        manifest,
        pl.DataFrame({"amount": [5.0], "plan": ["pro"], "surprise": [1]}),
    )

    assert result["schema"]["ok"] is True
    assert result["schema"]["unexpected"] == ["surprise"]
    assert result["schema"]["notes"]


def test_column_order_in_input_does_not_change_the_answer(tmp_path: Path) -> None:
    #порядок колонок в файле пользователя произволен; порядок в схеме — нет
    _make_artifact(tmp_path)
    pipeline, manifest = load_artifact(tmp_path)
    straight = pl.DataFrame({"amount": [5.0, 35.0], "plan": ["pro", "basic"]})
    swapped = straight.select(["plan", "amount"])

    first = predict_frame(pipeline, manifest, straight)
    second = predict_frame(pipeline, manifest, swapped)

    assert [row["prediction"] for row in first["predictions"]] == [
        row["prediction"] for row in second["predictions"]
    ]


def test_probabilities_are_labelled_by_task_class_order(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    pipeline, manifest = load_artifact(tmp_path)

    result = predict_frame(pipeline, manifest, pl.DataFrame({"amount": [35.0], "plan": ["pro"]}))
    probabilities = result["predictions"][0]["probabilities"]

    assert list(probabilities) == LABELS
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_predictions_export_carries_probability_columns(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    pipeline, manifest = load_artifact(tmp_path)
    result = predict_frame(
        pipeline, manifest, pl.DataFrame({"amount": [5.0, 35.0], "plan": ["pro", "basic"]})
    )

    table = predictions_to_frame(result)

    assert table.height == 2
    assert set(table.columns) == {"index", "prediction", "proba__no", "proba__yes"}


def test_schema_check_names_both_sides() -> None:
    schema = FeatureSchema(
        columns=["a", "b"], dtypes={}, numeric=["a"], categorical=["b"], datetime=[]
    )
    check = check_schema(["b", "c"], schema)

    assert check.ok is False
    assert check.missing == ["a"]
    assert check.unexpected == ["c"]
