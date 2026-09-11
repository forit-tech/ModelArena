"""Dataset Readiness: контекстность и содержательность находок.

Проверяется не «функция вернула список», а то, что вывод зависит от **тройки**
данные + задача + протокол, и что каждая находка объясняет последствие.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from backend.datasets.registry import DatasetRegistry, DatasetSnapshot
from backend.protocol.recommend import recommend_protocol
from backend.readiness.report import ReadinessReport, assess_readiness, estimate_effective_dimension
from backend.tasks.spec import TaskSpec, build_task_spec


@pytest.fixture
def snapshot(tmp_path: Path) -> DatasetSnapshot:
    registry = DatasetRegistry(root=tmp_path / "datasets")
    path = tmp_path / "seed.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}).write_parquet(path)
    return registry.import_file(path)


def assess(
    snapshot: DatasetSnapshot,
    frame: pl.DataFrame,
    *,
    target: str = "churned",
    task_type: str = "binary",
    positive_label: str | None = "True",
    group_column: str | None = None,
    time_column: str | None = None,
    features: list[str] | None = None,
    n_splits: int = 5,
) -> ReadinessReport:
    selected = features if features is not None else [
        name for name in frame.columns if name != target
    ]
    spec: TaskSpec = build_task_spec(
        frame,
        target_column=target,
        task_type=task_type,  # type: ignore[arg-type]
        feature_columns=selected,
        positive_label=positive_label if task_type == "binary" else None,
        group_column=group_column,
        time_column=time_column,
    )
    protocol = recommend_protocol(frame, spec, n_splits=n_splits)
    return assess_readiness(snapshot, frame, spec, protocol)


def healthy_frame(rows: int = 600) -> pl.DataFrame:
    #строки обязаны быть различны: полные дубли — сами по себе находка, и на них
    #проверка споткнётся раньше, чем на том, что тест хотел показать
    return pl.DataFrame(
        {
            "spend": [round(value * 1.37, 2) for value in range(rows)],
            "plan": [["basic", "pro", "enterprise"][value % 3] for value in range(rows)],
            "churned": [value % 3 == 0 for value in range(rows)],
        }
    )


# ---------------------------------------------------------------- статус


def test_healthy_dataset_is_ready(snapshot: DatasetSnapshot) -> None:
    report = assess(snapshot, healthy_frame())

    assert report.status == "READY"
    assert report.findings == []
    #сводка обязана быть содержательной: числа, а не «всё хорошо»
    assert "наблюдений на признак" in report.summary


def test_no_score_is_produced(snapshot: DatasetSnapshot) -> None:
    #балл прячет причину и провоцирует улучшать метрику вместо данных (D-22)
    payload = assess(snapshot, healthy_frame()).to_dict()

    assert "score" not in payload
    assert set(payload) == {"status", "summary", "context", "findings"}


def test_blocking_finding_gives_blocked_status(snapshot: DatasetSnapshot) -> None:
    #класс из одного объекта делает честное сравнение невозможным, а не просто плохим
    frame = pl.DataFrame(
        {
            "spend": [float(value) for value in range(300)],
            "churned": [True] * 2 + [False] * 298,
        }
    )
    report = assess(snapshot, frame, n_splits=5)

    assert report.status == "BLOCKED"
    blocking = {finding.code for finding in report.findings if finding.blocking}
    #два объекта на 300 строк — цель почти постоянна, сравнивать нечего
    assert "target_almost_constant" in blocking


def test_every_finding_explains_the_consequence(snapshot: DatasetSnapshot) -> None:
    frame = healthy_frame(60).with_columns(
        pl.Series("customer_id", [value // 3 for value in range(60)])
    )
    report = assess(snapshot, frame)

    assert report.findings
    for finding in report.findings:
        assert finding.explanation, finding.code
        assert finding.consequence, f"{finding.code}: не сказано, что исказится"
        assert finding.suggested_action, f"{finding.code}: не сказано, что делать"
        assert finding.threshold, f"{finding.code}: не назван порог"


# ---------------------------------------------------------------- контекстность


def test_identifier_is_a_risk_as_feature_and_a_structure_as_group(snapshot: DatasetSnapshot) -> None:
    """Одно и то же наблюдение означает разное в зависимости от роли колонки (D-22)."""
    frame = pl.DataFrame(
        {
            "customer_id": list(range(400)),
            "spend": [float(value % 89) for value in range(400)],
            "churned": [value % 3 == 0 for value in range(400)],
        }
    )

    as_feature = assess(snapshot, frame)
    assert as_feature.by_code("identifier_used_as_feature") is not None

    #та же колонка вне признаков — находки нет
    as_excluded = assess(snapshot, frame, features=["spend"])
    assert as_excluded.by_code("identifier_used_as_feature") is None


def test_missing_values_matter_only_in_selected_features(snapshot: DatasetSnapshot) -> None:
    """30% пропусков в неиспользуемой колонке не влияют ни на что."""
    rows = 400
    frame = pl.DataFrame(
        {
            "notes": [None if value % 2 else "текст" for value in range(rows)],
            "spend": [float(value % 89) for value in range(rows)],
            "churned": [value % 3 == 0 for value in range(rows)],
        }
    )

    with_notes = assess(snapshot, frame)
    without_notes = assess(snapshot, frame, features=["spend"])

    assert with_notes.by_code("feature_missing_values") is not None
    assert without_notes.by_code("feature_missing_values") is None


def test_time_column_is_a_problem_only_with_a_blind_protocol(snapshot: DatasetSnapshot) -> None:
    rows = 360
    frame = pl.DataFrame(
        {
            "created": [f"2024-{1 + value % 12:02d}-{1 + value % 28:02d}" for value in range(rows)],
            "spend": [float(value % 89) for value in range(rows)],
            "churned": [value % 3 == 0 for value in range(rows)],
        }
    )

    ignored = assess(snapshot, frame)
    honoured = assess(snapshot, frame, time_column="created", features=["spend"])

    assert ignored.by_code("time_structure_ignored") is not None
    assert ignored.status == "HIGH_RISK"
    #когда структура учтена, находки нет — сама по себе дата не является проблемой
    assert honoured.by_code("time_structure_ignored") is None


def test_thresholds_depend_on_sample_size(snapshot: DatasetSnapshot) -> None:
    """Один и тот же редкий класс читается по-разному при разном числе фолдов."""
    frame = pl.DataFrame(
        {
            "spend": [float(value) for value in range(300)],
            "churned": [True] * 12 + [False] * 288,
        }
    )

    few_folds = assess(snapshot, frame, n_splits=2)
    many_folds = assess(snapshot, frame, n_splits=10)

    def per_fold(report: ReadinessReport) -> float:
        finding = next(
            item for item in report.findings if item.code.startswith("minority_class_")
        )
        return float(finding.observed_value)

    #12 объектов на 2 фолда — по 6 на фолд; на 10 фолдов — по 1.2.
    #Порог не абсолютный: одно и то же число объектов читается по-разному
    assert per_fold(many_folds) < per_fold(few_folds)
    assert many_folds.status == "HIGH_RISK"


# ---------------------------------------------------------------- размерность


def test_effective_dimension_counts_encoded_columns_not_raw() -> None:
    #одна категориальная колонка на сорок уровней даёт сорок столбцов, а не один
    frame = pl.DataFrame(
        {
            "numeric": [float(value) for value in range(120)],
            "category": [f"c{value % 40}" for value in range(120)],
        }
    )

    assert estimate_effective_dimension(frame, ["numeric"]) == 1
    assert estimate_effective_dimension(frame, ["numeric", "category"]) == 41


def test_wide_dataset_is_flagged_even_with_few_columns(snapshot: DatasetSnapshot) -> None:
    #100 строк и одна категориальная колонка на 50 уровней: колонок две, измерений 51
    rows = 100
    frame = pl.DataFrame(
        {
            "category": [f"c{value % 50}" for value in range(rows)],
            "churned": [value % 3 == 0 for value in range(rows)],
        }
    )
    report = assess(snapshot, frame)

    finding = report.by_code("sample_size_vs_dimension")
    assert finding is not None
    assert finding.severity in {"caution", "high_risk"}
    assert "после кодирования" in finding.explanation


# ---------------------------------------------------------------- регрессия


def test_constant_regression_target_blocks(snapshot: DatasetSnapshot) -> None:
    frame = pl.DataFrame(
        {
            "spend": [float(value) for value in range(100)],
            "price": [7.0] * 100,
        }
    )
    spec = build_task_spec(
        frame, target_column="price", task_type="regression", feature_columns=["spend"]
    )
    protocol = recommend_protocol(frame, spec)
    report = assess_readiness(snapshot, frame, spec, protocol)

    assert report.status == "BLOCKED"
    assert report.by_code("regression_target_constant") is not None


def test_snapshot_field_is_not_required_to_change_the_verdict(snapshot: DatasetSnapshot) -> None:
    #вывод зависит от данных, задачи и протокола, а не от метаданных снимка
    frame = healthy_frame()
    renamed = replace(snapshot, name="другое имя")

    assert assess(snapshot, frame).status == assess(renamed, frame).status
