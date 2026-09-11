"""Leakage Guard: ловит механизмы утечки и **молчит там, где утечки нет**.

Отрицательный набор здесь важнее положительного. Детектор, кричащий на всё, бесполезен:
доверять можно только тому, чьё молчание тоже что-то значит (D-23). Поэтому на каждый
положительный сценарий есть парный честный, который обязан пройти чисто.
"""
from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from backend.leakage.report import LeakageReport, inspect_leakage
from backend.protocol.folds import build_folds
from backend.protocol.recommend import recommend_protocol
from backend.tasks.spec import build_task_spec

ROWS = 300


def inspect(
    frame: pl.DataFrame,
    *,
    target: str = "churned",
    task_type: str = "binary",
    positive_label: str | None = "True",
    features: list[str] | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    n_splits: int = 5,
) -> LeakageReport:
    selected = features if features is not None else [
        name for name in frame.columns if name != target
    ]
    spec = build_task_spec(
        frame,
        target_column=target,
        task_type=task_type,  # type: ignore[arg-type]
        feature_columns=selected,
        positive_label=positive_label if task_type == "binary" else None,
        group_column=group_column,
        time_column=time_column,
    )
    protocol = recommend_protocol(frame, spec, n_splits=n_splits)
    folds = build_folds(frame, spec, protocol)
    return inspect_leakage(frame, spec, protocol, folds)


def honest_frame(rows: int = ROWS) -> pl.DataFrame:
    """Датасет без утечки: сигнал есть, но он честный и неполный."""
    return pl.DataFrame(
        {
            "spend": [round(value * 1.31, 2) for value in range(rows)],
            "plan": [["basic", "pro", "enterprise"][value % 3] for value in range(rows)],
            "tenure": [value % 37 for value in range(rows)],
            "churned": [value % 4 == 0 for value in range(rows)],
        }
    )


# ============================================================ положительные


def test_feature_equal_to_target_is_structural_and_blocking() -> None:
    frame = honest_frame().with_columns(pl.col("churned").alias("copy_of_target"))
    report = inspect(frame)

    signal = report.by_code("target_copy_among_features")
    assert signal is not None
    assert signal.evidence_level == "structural"
    assert signal.blocking
    assert report.risk == "confirmed"
    #механизм обязан быть назван в терминах сравнения моделей, а не «это плохо»
    assert "leaderboard" in signal.mechanism.lower()


def test_string_encoding_of_target_is_caught() -> None:
    frame = honest_frame().with_columns(
        pl.when(pl.col("churned")).then(pl.lit("ушёл")).otherwise(pl.lit("остался")).alias("status")
    )
    report = inspect(frame)

    #строка не совпадает с целью посимвольно, но восстанавливает её полностью
    assert report.by_code("single_feature_reconstructs_target") is not None
    assert report.risk in {"high", "confirmed"}


def test_regression_target_derived_feature_is_structural() -> None:
    rows = 200
    frame = pl.DataFrame(
        {
            "feature": [float(value % 53) for value in range(rows)],
            "price": [float(value) * 2.5 + 7 for value in range(rows)],
        }
    ).with_columns((pl.col("price") * 0.2 - 3).alias("price_share"))

    report = inspect(
        frame, target="price", task_type="regression", positive_label=None, n_splits=3
    )
    signal = report.by_code("target_derived_feature")

    assert signal is not None
    assert signal.blocking
    assert "price_share" in signal.columns


def test_target_plus_noise_is_caught_by_measurement_not_by_correlation() -> None:
    rows = 300
    frame = pl.DataFrame(
        {
            "noise": [float(value % 17) for value in range(rows)],
            "price": [float(value % 91) for value in range(rows)],
        }
    ).with_columns((pl.col("price") + pl.Series([0.001 * (value % 3) for value in range(rows)])).alias("almost_price"))

    report = inspect(frame, target="price", task_type="regression", positive_label=None, n_splits=3)

    #точного восстановления нет — значит structural-проверка молчит,
    #а измерение вне обучающей части сигнал даёт
    assert report.by_code("target_derived_feature") is None
    assert report.by_code("single_feature_reconstructs_target") is not None


def test_entity_crossing_is_reported_when_protocol_ignores_groups() -> None:
    """Сущность выбрана, но протокол её не изолирует — строки одного клиента разойдутся.

    Тест намеренно строит рассогласованную пару: так бывает, когда пользователь
    переопределяет стратегию вручную, оставив колонку группировки.
    """
    frame = honest_frame().with_columns(
        pl.Series("customer_id", [value // 3 for value in range(ROWS)])
    )
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend", "plan", "tenure"],
        positive_label="True",
        group_column="customer_id",
    )
    #протокол принудительно не групп-зависимый
    protocol = replace(
        recommend_protocol(frame, spec, n_splits=5),
        cv_splitter="stratified_k_fold",
        holdout_strategy="stratified",
    )
    folds = build_folds(frame, spec, protocol)
    report = inspect_leakage(frame, spec, protocol, folds)

    signal = report.by_code("group_crosses_fold_boundary")
    assert signal is not None, "сущности пересекли границу, но сигнала нет"
    assert signal.blocking
    assert signal.evidence_level == "structural"
    assert report.risk == "confirmed"


def test_duplicate_rows_crossing_folds_are_reported() -> None:
    base = honest_frame(60)
    frame = pl.concat([base, base, base, base, base])
    report = inspect(frame, n_splits=5)

    signal = report.by_code("duplicate_rows_cross_folds")
    assert signal is not None
    assert signal.evidence_level == "structural"
    assert signal.blocking
    assert signal.observed_value["crossing_validation_rows"] > 0


def test_missingness_encoding_target_is_caught() -> None:
    rows = 400
    churned = [value % 4 == 0 for value in range(rows)]
    frame = pl.DataFrame(
        {
            "spend": [round(value * 1.13, 2) for value in range(rows)],
            #поле заполняется только для тех, кто не ушёл: значение безобидно, пропуск несёт ответ
            "closing_note": [None if flag else f"note-{value}" for value, flag in enumerate(churned)],
            "churned": churned,
        }
    )
    report = inspect(frame)

    signal = report.by_code("missingness_encodes_target")
    assert signal is not None
    assert "пропуск" in signal.mechanism.lower()


def test_post_outcome_name_is_heuristic_and_never_blocks() -> None:
    frame = honest_frame().rename({"tenure": "refund_reason_code"})
    report = inspect(frame)

    signal = report.by_code("post_outcome_naming")
    assert signal is not None
    assert signal.evidence_level == "heuristic"
    #подозрение по имени не имеет права блокировать: колонка может быть законной
    assert not signal.blocking
    assert report.risk == "suspected"


# ============================================================ отрицательные


def test_clean_dataset_produces_no_signals() -> None:
    report = inspect(honest_frame())

    assert report.signals == []
    assert report.risk == "none"
    assert "не обнаружено" in report.summary


def test_strong_but_honest_predictor_is_not_leakage() -> None:
    """Сильный честный признак существует и не должен объявляться утечкой.

    Здесь `tenure` объясняет цель заметно лучше случайного, но не полностью:
    остаётся собственная дисперсия, и точного восстановления нет.
    """
    rows = 400
    tenure = [value % 40 for value in range(rows)]
    #цель зависит от признака, но не определяется им: у каждого значения есть оба исхода
    churned = [(value % 40) < 10 if value % 7 else (value % 40) >= 10 for value in range(rows)]
    frame = pl.DataFrame(
        {
            "tenure": tenure,
            "spend": [round(value * 0.9, 2) for value in range(rows)],
            "churned": churned,
        }
    )
    report = inspect(frame)

    assert report.by_code("target_copy_among_features") is None
    assert report.by_code("single_feature_reconstructs_target") is None


def test_ordinary_identifier_is_not_leakage_by_itself() -> None:
    #ID сам по себе — вопрос Readiness, а не Leakage Guard: механизма утечки в нём нет
    frame = honest_frame().with_columns(pl.Series("customer_id", list(range(ROWS))))
    report = inspect(frame)

    assert report.by_code("target_copy_among_features") is None
    assert report.by_code("duplicate_rows_cross_folds") is None


def test_group_aware_protocol_removes_the_group_finding() -> None:
    """Если группы действительно изолированы — находки нет."""
    rows = 300
    frame = pl.DataFrame(
        {
            "customer_id": [value // 3 for value in range(rows)],
            "spend": [round(value * 1.7, 2) for value in range(rows)],
            "churned": [(value // 3) % 4 == 0 for value in range(rows)],
        }
    )
    report = inspect(frame, group_column="customer_id", features=["spend"], n_splits=4)

    assert report.by_code("group_crosses_fold_boundary") is None


def test_legitimate_time_feature_with_time_protocol_is_clean() -> None:
    rows = 300
    frame = pl.DataFrame(
        {
            #даты растут вместе с номером строки: именно так выглядит законный
            #временной признак. Циклические даты нарушают порядок по построению
            "created": [f"2024-{1 + value // 26:02d}-{1 + value % 26:02d}" for value in range(rows)],
            "spend": [round(value * 1.7, 2) for value in range(rows)],
            "churned": [value % 4 == 0 for value in range(rows)],
        }
    )
    report = inspect(frame, time_column="created", features=["spend"], n_splits=3)

    assert report.by_code("training_after_validation_in_time") is None


def test_high_cardinality_category_is_not_leakage() -> None:
    rows = 400
    frame = pl.DataFrame(
        {
            "city": [f"city-{value % 120}" for value in range(rows)],
            "spend": [round(value * 1.7, 2) for value in range(rows)],
            "churned": [value % 4 == 0 for value in range(rows)],
        }
    )
    report = inspect(frame)

    assert report.signals == []


def test_imbalance_without_leakage_is_clean() -> None:
    rows = 400
    frame = pl.DataFrame(
        {
            "spend": [round(value * 1.7, 2) for value in range(rows)],
            "plan": [["basic", "pro"][value % 2] for value in range(rows)],
            "churned": [value % 25 == 0 for value in range(rows)],
        }
    )
    report = inspect(frame, n_splits=3)

    assert report.by_code("single_feature_reconstructs_target") is None
    assert report.risk == "none"


def test_missingness_unrelated_to_target_is_clean() -> None:
    rows = 400
    frame = pl.DataFrame(
        {
            "spend": [round(value * 1.7, 2) for value in range(rows)],
            #пропуск расставлен по позиции строки, а не по исходу
            "comment": [None if value % 3 else f"c{value}" for value in range(rows)],
            "churned": [value % 4 == 0 for value in range(rows)],
        }
    )
    report = inspect(frame)

    assert report.by_code("missingness_encodes_target") is None


# ============================================================ типы задач


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_target_copy_is_caught_for_every_classification_type(task: str) -> None:
    rows = 300
    values = (
        [value % 2 == 0 for value in range(rows)]
        if task == "binary"
        else [f"class-{value % 4}" for value in range(rows)]
    )
    frame = pl.DataFrame(
        {
            "spend": [round(value * 1.7, 2) for value in range(rows)],
            "target": values,
        }
    ).with_columns(pl.col("target").alias("shadow"))

    report = inspect(
        frame,
        target="target",
        task_type=task,
        positive_label="True" if task == "binary" else None,
        n_splits=3,
    )

    assert report.by_code("target_copy_among_features") is not None
    assert report.risk == "confirmed"


def test_disclaimer_never_claims_certainty() -> None:
    report = inspect(honest_frame())

    #инструмент не может доказать отсутствие утечки, и обязан это признавать
    assert "не может доказать" in report.disclaimer


# ============================================================ связь с готовностью


def test_confirmed_leakage_makes_readiness_blocked(tmp_path) -> None:
    """READY при подтверждённой блокирующей утечке невозможно по построению (D-23)."""
    from backend.datasets.registry import DatasetRegistry
    from backend.readiness.report import assess_readiness

    registry = DatasetRegistry(root=tmp_path / "datasets")
    seed = tmp_path / "seed.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}).write_parquet(seed)
    snapshot = registry.import_file(seed)

    frame = honest_frame().with_columns(pl.col("churned").alias("copy_of_target"))
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend", "plan", "tenure", "copy_of_target"],
        positive_label="True",
    )
    protocol = recommend_protocol(frame, spec)
    folds = build_folds(frame, spec, protocol)
    leakage = inspect_leakage(frame, spec, protocol, folds)

    without = assess_readiness(snapshot, frame, spec, protocol)
    with_leakage = assess_readiness(snapshot, frame, spec, protocol, leakage)

    #без учёта утечки данные выглядят приемлемо, с учётом — сравнение невозможно
    assert without.status != "BLOCKED"
    assert with_leakage.status == "BLOCKED"
    assert with_leakage.by_code("leakage.target_copy_among_features") is not None


def test_heuristic_leakage_does_not_raise_readiness_status(tmp_path) -> None:
    from backend.datasets.registry import DatasetRegistry
    from backend.readiness.report import assess_readiness

    registry = DatasetRegistry(root=tmp_path / "datasets")
    seed = tmp_path / "seed.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}).write_parquet(seed)
    snapshot = registry.import_file(seed)

    frame = honest_frame().rename({"tenure": "refund_reason_code"})
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["spend", "plan", "refund_reason_code"],
        positive_label="True",
    )
    protocol = recommend_protocol(frame, spec)
    folds = build_folds(frame, spec, protocol)
    leakage = inspect_leakage(frame, spec, protocol, folds)
    report = assess_readiness(snapshot, frame, spec, protocol, leakage)

    finding = report.by_code("leakage.post_outcome_naming")
    assert finding is not None
    #подозрение по имени не поднимает статус: иначе любая колонка со словом «причина»
    #блокировала бы работу
    assert finding.severity == "info"
    assert report.status != "BLOCKED"


# ============================================================ «не смог проверить»


def test_unevaluated_check_is_reported_not_treated_as_clean() -> None:
    """L-05. Молчание сломанной проверки не имеет права выглядеть как чистый результат."""
    report = inspect(honest_frame(36), n_splits=2)

    codes = {item.check for item in report.not_evaluated}
    assert "single_feature_reconstructs_target" in codes
    assert not report.fully_evaluated
    assert "не удалось" in report.summary


def test_regression_missingness_is_declared_unevaluated() -> None:
    rows = 200
    frame = pl.DataFrame(
        {
            "spend": [float(value % 71) for value in range(rows)],
            "note": [None if value % 3 else f"n{value}" for value in range(rows)],
            "price": [float(value % 53) for value in range(rows)],
        }
    )
    report = inspect(frame, target="price", task_type="regression", positive_label=None, n_splits=3)

    missing = next(
        item for item in report.not_evaluated if item.check == "missingness_encodes_target"
    )
    assert "регрессии" in missing.reason


def test_probe_budget_names_unchecked_columns() -> None:
    """L-03. Раньше признаки за пределом среза не проверялись и об этом никто не знал."""
    rows = 200
    data = {
        f"f{index}": [float((value * (index + 3)) % 47) for value in range(rows)]
        for index in range(90)
    }
    data["churned"] = [value % 4 == 0 for value in range(rows)]
    report = inspect(pl.DataFrame(data), n_splits=5)

    exhausted = [
        item
        for item in report.not_evaluated
        if item.check == "single_feature_reconstructs_target" and "бюджет" in item.reason
    ]
    assert exhausted, "бюджет исчерпан, но непроверенные признаки не названы"
    assert exhausted[0].scope.startswith("column:")


def test_unencodable_column_is_named() -> None:
    rows = 120
    frame = pl.DataFrame(
        {
            "spend": [float(value % 61) for value in range(rows)],
            "constant": ["same"] * rows,
            "churned": [value % 4 == 0 for value in range(rows)],
        }
    )
    report = inspect(frame, n_splits=3)

    assert any(item.scope == "column:constant" for item in report.not_evaluated)


def test_boundary_check_scales_linearly() -> None:
    """L-06. Множество обучающих ключей строилось внутри генератора и пересоздавалось
    на каждой строке holdout: проверка была квадратичной и на 50 000 строк занимала минуту.

    Тест сравнивает не абсолютное время, а рост: удвоение объёма не должно давать
    больше чем троекратное замедление.
    """
    import time

    from backend.leakage.checks import boundaries

    def measure(rows: int) -> float:
        frame = pl.DataFrame(
            {
                "spend": [float(value % 997) for value in range(rows)],
                "plan": [["basic", "pro"][value % 2] for value in range(rows)],
                "churned": [value % 7 == 0 for value in range(rows)],
            }
        )
        spec = build_task_spec(
            frame,
            target_column="churned",
            task_type="binary",
            feature_columns=["spend", "plan"],
            positive_label="True",
        )
        protocol = recommend_protocol(frame, spec, n_splits=3)
        folds = build_folds(frame, spec, protocol)
        start = time.perf_counter()
        boundaries.check(frame, spec, protocol, folds)
        return time.perf_counter() - start

    small = measure(8_000)
    large = measure(16_000)

    #квадратичный рост дал бы четырёхкратное замедление и выше
    assert large < max(small * 3, 0.5), f"рост {large / max(small, 1e-6):.1f}x — похоже на квадратичный"
