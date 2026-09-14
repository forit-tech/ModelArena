"""Диагностика: вывод только из измерений.

Проверяется главное свойство слоя — он не имеет права сказать «переобучена», не показав
чисел, и не имеет права промолчать, когда мерить нечем. Отсутствие замера на обучающей
части — это «разрыв не измерен», а не «разрыва нет».
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backend.arena.predictions import SPLIT_CV, SPLIT_HOLDOUT, SPLIT_TRAIN_PROBE
from backend.diagnostics import thresholds
from backend.diagnostics.errors import select_error_rows
from backend.diagnostics.report import build_diagnostics

LABELS = ["no", "yes"]
PER_FOLD = 40


def _rows(split: str, fold: int, accuracy: float, start: int) -> list[dict]:
    correct = round(accuracy * PER_FOLD)
    prepared = []

    for index in range(PER_FOLD):
        truth = LABELS[index % 2]
        predicted = truth if index < correct else LABELS[(index + 1) % 2]
        confidence = 0.9 if predicted == "yes" else 0.1
        prepared.append(
            {
                "__row_id__": start + index,
                "split": split,
                "fold": fold,
                "y_true": truth,
                "y_pred": predicted,
                "proba__no": 1.0 - confidence,
                "proba__yes": confidence,
            }
        )

    return prepared


def _frame(
    validation: list[float], probe: list[float] | None = None, holdout: float | None = None
) -> pl.DataFrame:
    rows: list[dict] = []
    cursor = 0

    for fold, accuracy in enumerate(validation):
        rows.extend(_rows(SPLIT_CV, fold, accuracy, cursor))
        cursor += PER_FOLD

    for fold, accuracy in enumerate(probe or []):
        rows.extend(_rows(SPLIT_TRAIN_PROBE, fold, accuracy, cursor))
        cursor += PER_FOLD

    if holdout is not None:
        rows.extend(_rows(SPLIT_HOLDOUT, -1, holdout, cursor))

    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.col("fold").cast(pl.Int32),
        pl.col("proba__no").cast(pl.Float32),
        pl.col("proba__yes").cast(pl.Float32),
    )


def _diagnostics(frame: pl.DataFrame, metric: str = "accuracy"):
    return build_diagnostics(
        frame,
        contender_key="model",
        task_type="binary",
        class_labels=LABELS,
        positive_label="yes",
        metric=metric,
    )


# ---------------------------------------------------------------- разрыв


def test_gap_is_measured_and_carries_its_numbers() -> None:
    """Предупреждение о переобучении обязано показывать числа, а не ощущение."""
    diagnostics = _diagnostics(_frame([0.70, 0.70, 0.70], probe=[0.98, 0.98, 0.98]))
    finding = next(item for item in diagnostics.findings if item.code == "train_validation_gap")

    assert finding.severity == "high"
    assert finding.evidence["train_mean"] == pytest.approx(0.975, abs=0.05)
    assert finding.evidence["validation_mean"] == pytest.approx(0.70, abs=0.05)
    assert finding.evidence["gap"] > 0
    #порог назван рядом с измерением, а не спрятан в интерфейсе
    assert finding.evidence["relative_threshold"] == thresholds.RELATIVE_GAP_HIGH
    assert "0." in finding.explanation


def test_without_a_train_probe_the_gap_is_not_guessed() -> None:
    """«Разрыв не измерен» и «разрыва нет» — разные утверждения."""
    diagnostics = _diagnostics(_frame([0.70, 0.72, 0.71]))

    assert diagnostics.mean_gap is None
    assert all(row.gap is None for row in diagnostics.folds)
    assert not [item for item in diagnostics.findings if item.code == "train_validation_gap"]
    assert any("измерить нечем" in note for note in diagnostics.notes)


def test_small_gap_does_not_raise_a_warning() -> None:
    diagnostics = _diagnostics(_frame([0.80, 0.80, 0.80], probe=[0.82, 0.82, 0.82]))

    assert not [item for item in diagnostics.findings if item.code == "train_validation_gap"]


def test_validation_better_than_train_is_not_overfitting() -> None:
    #проверочная часть лучше обучающей — следствие регуляризации или разбиения,
    #а не признак беды
    assert thresholds.gap_severity(metric="accuracy", gap=-0.2, train_score=0.7) == "info"


# ---------------------------------------------------------------- устойчивость


def test_fold_instability_is_reported_with_per_fold_numbers() -> None:
    diagnostics = _diagnostics(_frame([0.95, 0.50, 0.90, 0.55]))
    finding = next(item for item in diagnostics.findings if item.code == "fold_instability")

    assert finding.severity in {"caution", "high"}
    assert len(finding.evidence["per_fold"]) == 4
    assert finding.evidence["threshold"] in {
        thresholds.FOLD_SPREAD_CAUTION,
        thresholds.FOLD_SPREAD_HIGH,
    }


def test_stable_folds_produce_no_instability_warning() -> None:
    diagnostics = _diagnostics(_frame([0.80, 0.81, 0.80, 0.79]))

    assert not [item for item in diagnostics.findings if item.code == "fold_instability"]


# ---------------------------------------------------------------- классификация


def test_classification_diagnostics_report_per_class_and_confusion() -> None:
    diagnostics = _diagnostics(_frame([0.70, 0.70, 0.70]))
    classification = diagnostics.classification

    assert classification["available"] is True
    assert classification["is_binary"] is True
    assert {item["label"] for item in classification["per_class"]} == set(LABELS)
    assert len(classification["confusion"]["rows"]) == 2
    assert sum(sum(row) for row in classification["confusion"]["rows"]) == 3 * PER_FOLD


def test_model_without_probabilities_gets_not_applicable_not_zero() -> None:
    """Отсутствие вероятностей — «неприменимо», а не нулевая калибровка."""
    frame = _frame([0.70, 0.70]).drop("proba__no", "proba__yes")
    diagnostics = _diagnostics(frame)

    confidence = diagnostics.classification["confidence"]
    assert confidence["available"] is False
    assert "вероятност" in confidence["reason"]


def test_regression_diagnostics_do_not_pretend_to_be_classification() -> None:
    rows = 120
    generator = np.random.default_rng(5)
    truth = generator.normal(100, 20, size=rows)
    frame = pl.DataFrame(
        {
            "__row_id__": np.arange(rows, dtype=np.int64),
            "split": [SPLIT_CV] * rows,
            "fold": np.zeros(rows, dtype=np.int32),
            "y_true": truth,
            "y_pred": truth + generator.normal(5, 3, size=rows),
        }
    )
    diagnostics = build_diagnostics(
        frame,
        contender_key="model",
        task_type="regression",
        class_labels=[],
        positive_label=None,
        metric="rmse",
    )

    assert diagnostics.classification is None
    assert diagnostics.regression["available"] is True
    #систематическое смещение заметно в остатках и не видно в RMSE
    assert diagnostics.regression["residuals"]["systematic_bias"] is True
    assert diagnostics.regression["by_target_range"]
    assert diagnostics.regression["largest_errors"]


# ---------------------------------------------------------------- разбор ошибок


def test_error_rows_separate_features_from_the_target(tmp_path, monkeypatch) -> None:
    """Целевая колонка не должна быть подписана словом «признак»."""
    from backend.core.config import get_settings
    from backend.datasets.registry import DatasetRegistry

    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    rows = 3 * PER_FOLD
    source = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            "amount": [float(index) for index in range(rows)],
            "plan": [["basic", "pro"][index % 2] for index in range(rows)],
            "label": [LABELS[index % 2] for index in range(rows)],
        }
    ).write_parquet(source)

    registry = DatasetRegistry()
    snapshot = registry.import_file(source, name="demo")

    result = select_error_rows(
        _frame([0.70, 0.70, 0.70]),
        dataset_id=snapshot.dataset_id,
        task_type="binary",
        class_labels=LABELS,
        positive_label="yes",
        feature_columns=["amount", "plan"],
        kind="all_mistakes",
        limit=5,
    )

    assert result["total_matching"] > 0
    row = result["rows"][0]
    assert set(row["features"]) == {"amount", "plan"}
    #цель уходит в контекст, а не в признаки
    assert "label" in row["context"]
    assert "label" not in row["features"]
    #идентификатор строки — адрес наблюдения, и признаком он не становится
    assert "__row_id__" not in row["features"]
    assert isinstance(row["row_id"], int)

    get_settings.cache_clear()
