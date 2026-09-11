"""Реестр адаптеров: статусы с причинами, baseline, потоки из бюджета.

Главное, что здесь проверяется, — адаптер **никогда не исчезает молча**. Пустое место
в списке контендеров читается как «модель проиграла», а не как «её не запускали».
"""
from __future__ import annotations

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from backend.core.errors import ValidationError
from backend.models.adapters import core
from backend.models.context import DatasetContext, ResourceBudget, build_context
from backend.models.registry import (
    adapters_for,
    build_contenders,
    build_estimator,
)
from backend.preprocessing.profiles import (
    build_preprocessor,
    normalize_categoricals,
    split_feature_types,
)
from backend.tasks.spec import build_task_spec

BUDGET = ResourceBudget(max_parallel_fits=2, threads_per_fit=3, memory_budget_mb=4096)


def context_for(
    task_type: str = "binary",
    *,
    n_rows: int = 5_000,
    n_train_rows: int = 4_000,
    effective_dimension: int = 12,
    max_cardinality: int = 4,
) -> DatasetContext:
    return DatasetContext(
        task_type=task_type,
        n_rows=n_rows,
        n_train_rows=n_train_rows,
        n_features=6,
        effective_dimension=effective_dimension,
        n_numeric=4,
        n_categorical=2,
        n_datetime=0,
        max_cardinality=max_cardinality,
        missing_ratio=0.0,
        has_missing=False,
        n_classes=2 if task_type != "regression" else None,
        min_class_count=400 if task_type != "regression" else None,
    )


# ---------------------------------------------------------------- состав


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "regression"])
def test_every_task_has_a_baseline(task_type: str) -> None:
    contenders = build_contenders(context_for(task_type))

    baselines = [item for item in contenders if item.is_baseline]
    assert len(baselines) == 1
    assert baselines[0].status == "ready"


def test_baseline_cannot_be_deselected() -> None:
    #без точки отсчёта метрика 0.88 ничего не значит, поэтому выбор пользователя
    #сужает набор, но baseline не убирает
    contenders = build_contenders(context_for(), selected_keys=["logistic_regression"])

    keys = {item.adapter_key for item in contenders}
    assert keys == {"baseline_majority", "logistic_regression"}


def test_selecting_nothing_runnable_is_refused() -> None:
    with pytest.raises(ValidationError, match="кроме baseline"):
        build_contenders(context_for(), selected_keys=["unknown_model"])


# ---------------------------------------------------------------- статусы с причинами


def test_unavailable_adapter_stays_visible_with_an_install_hint() -> None:
    """Недоступный бустер остаётся в списке: исчезнув, он выглядел бы проигравшим."""
    contenders = {item.adapter_key: item for item in build_contenders(context_for())}

    for key in ("xgboost", "lightgbm", "catboost"):
        contender = contenders[key]

        if contender.status == "unavailable":
            assert "не установлен" in contender.reason
            assert "pip install" in contender.reason
            assert not contender.runnable


def test_svm_is_skipped_on_large_data_with_a_reason() -> None:
    contenders = {item.adapter_key: item for item in build_contenders(context_for(n_train_rows=50_000))}
    svm = contenders["svm_rbf"]

    assert svm.status == "skipped"
    assert "квадрат" in svm.reason
    assert not svm.runnable


def test_knn_is_skipped_on_many_rows_and_cautioned_in_high_dimension() -> None:
    huge = {item.adapter_key: item for item in build_contenders(context_for(n_rows=300_000))}
    assert huge["knn"].status == "skipped"
    assert "inference" in huge["knn"].reason

    wide = {
        item.adapter_key: item
        for item in build_contenders(context_for(effective_dimension=120))
    }
    assert wide["knn"].status == "caution"
    assert "размерности" in wide["knn"].reason


def test_every_non_ready_status_carries_a_reason() -> None:
    for task_type in ("binary", "multiclass", "regression"):
        for contender in build_contenders(context_for(task_type, n_train_rows=50_000)):
            if contender.status != "ready":
                assert contender.reason, f"{contender.adapter_key}: статус без причины"


# ---------------------------------------------------------------- сборка estimator


def test_threads_come_from_the_budget_not_from_the_model() -> None:
    """`n_jobs=-1` при параллельном запуске даёт «модели × все ядра» (D-12)."""
    contenders = {item.adapter_key: item for item in build_contenders(context_for())}
    estimator = build_estimator("binary", contenders["random_forest"], BUDGET, seed=42)

    assert estimator.n_jobs == BUDGET.threads_per_fit
    assert estimator.n_jobs != -1
    assert estimator.random_state == 42


def test_no_core_adapter_hardcodes_all_cores() -> None:
    #защита от возвращения дефекта: ни один адаптер не имеет права ставить n_jobs=-1
    narrow = ResourceBudget(max_parallel_fits=4, threads_per_fit=1, memory_budget_mb=1024)

    for task_type in ("binary", "regression"):
        for contender in build_contenders(context_for(task_type)):
            if not contender.runnable:
                continue

            estimator = build_estimator(task_type, contender, narrow, seed=7)
            n_jobs = getattr(estimator, "n_jobs", None)
            assert n_jobs != -1, f"{contender.adapter_key} требует все ядра"


def test_baseline_ignores_features_entirely() -> None:
    contenders = {item.adapter_key: item for item in build_contenders(context_for())}
    baseline = contenders["baseline_majority"]

    assert baseline.preprocessing_profile == "passthrough_dummy"
    assert baseline.estimated_cost == 0.0


def test_unrunnable_contender_refuses_to_build() -> None:
    contenders = {item.adapter_key: item for item in build_contenders(context_for(n_train_rows=50_000))}

    with pytest.raises(ValidationError, match="skipped"):
        build_estimator("binary", contenders["svm_rbf"], BUDGET, seed=1)


def test_logistic_regression_does_not_silently_balance_classes() -> None:
    """Балансировка меняет калибровку и делает несравнимыми log loss и PR-AUC (D-12)."""
    contenders = {item.adapter_key: item for item in build_contenders(context_for())}

    assert "class_weight" not in contenders["logistic_regression"].params


# ---------------------------------------------------------------- профили препроцессинга


def test_families_get_different_profiles() -> None:
    #честность задаётся протоколом, а не одинаковым кодированием (D-7)
    profiles = {
        item.adapter_key: item.preprocessing_profile for item in build_contenders(context_for())
    }

    assert profiles["logistic_regression"] == "scaled_onehot"
    assert profiles["random_forest"] == "tree_ordinal"
    assert profiles["hist_gradient_boosting"] == "native_missing"


def test_preprocessor_fits_only_on_what_it_is_given() -> None:
    """Статистики считаются по переданной части, а не по всему датасету.

    Медиана, посчитанная на обучающей части, обязана отличаться от медианы всего
    датасета — иначе трансформер увидел бы проверочные строки.
    """
    import pandas as pd

    frame = pd.DataFrame({"value": [float(index) for index in range(100)]})
    types = split_feature_types(frame)

    early = build_preprocessor("scaled_onehot", types).fit(frame.iloc[:50])
    whole = build_preprocessor("scaled_onehot", types).fit(frame)

    early_median = early.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
    whole_median = whole.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]

    assert early_median != whole_median


def test_native_missing_profile_keeps_nulls_for_the_model() -> None:
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"value": [1.0, None, 3.0, 4.0]})
    types = split_feature_types(frame)
    transformed = build_preprocessor("native_missing", types).fit_transform(frame)

    #HistGradientBoosting работает с пропусками сам; подстановка стёрла бы информацию
    assert np.isnan(transformed).any()


def test_scaled_profile_fills_and_scales() -> None:
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"value": [1.0, None, 3.0, 4.0]})
    types = split_feature_types(frame)
    transformed = build_preprocessor("scaled_onehot", types).fit_transform(frame)

    assert not np.isnan(transformed).any()


def test_unknown_category_does_not_become_the_most_frequent_one() -> None:
    """Неизвестная категория получает значение вне диапазона кодов, а не самую частую."""
    import pandas as pd

    train = pd.DataFrame({"plan": ["basic", "basic", "pro"]})
    unseen = pd.DataFrame({"plan": ["enterprise"]})
    types = split_feature_types(train)
    encoder = build_preprocessor("tree_ordinal", types).fit(train)

    assert encoder.transform(unseen)[0][0] == -1


def test_bool_is_treated_as_a_category_not_a_number() -> None:
    import pandas as pd

    frame = pd.DataFrame({"flag": [True, False, True], "amount": [1.0, 2.0, 3.0]})
    types = split_feature_types(frame)

    assert types.categorical == ["flag"]
    assert types.numeric == ["amount"]


def test_categorical_normalization_unifies_representations() -> None:
    import pandas as pd

    frame = pd.DataFrame({"flag": [True, "True", None]})
    normalized = normalize_categoricals(frame, ["flag"])

    #True и "True" из разных источников не должны стать разными категориями
    assert normalized["flag"].tolist()[:2] == ["True", "True"]
    assert normalized["flag"].tolist()[2] is None


# ---------------------------------------------------------------- контекст


def test_context_counts_encoded_dimension_not_columns() -> None:
    rows = 200
    frame = pl.DataFrame(
        {
            "amount": [float(value) for value in range(rows)],
            "city": [f"c{value % 30}" for value in range(rows)],
            "churned": [value % 4 == 0 for value in range(rows)],
        }
    )
    spec = build_task_spec(
        frame,
        target_column="churned",
        task_type="binary",
        feature_columns=["amount", "city"],
        positive_label="True",
    )
    context = build_context(frame, spec, n_train_rows=160)

    assert context.n_features == 2
    assert context.effective_dimension == 31
    assert context.max_cardinality == 30


def test_budget_never_hands_out_every_core() -> None:
    budget = ResourceBudget.detect()

    assert budget.threads_per_fit >= 1
    assert budget.max_parallel_fits * budget.threads_per_fit <= max(1, (__import__("os").cpu_count() or 2))


def test_pipeline_assembles_for_every_runnable_contender() -> None:
    """Каждый запускаемый контендер обязан собираться в реальный Pipeline."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "amount": [float(value) for value in range(60)],
            "plan": [["basic", "pro"][value % 2] for value in range(60)],
        }
    )
    types = split_feature_types(frame)

    for contender in build_contenders(context_for()):
        if not contender.runnable:
            continue

        pipeline = Pipeline(
            [
                ("preprocessing", build_preprocessor(contender.preprocessing_profile, types)),
                ("model", build_estimator("binary", contender, BUDGET, seed=0)),
            ]
        )
        assert isinstance(pipeline, Pipeline), contender.adapter_key


def test_adapter_keys_are_unique_within_a_task() -> None:
    for task_type in ("binary", "multiclass", "regression"):
        keys = [adapter.key for adapter in adapters_for(task_type)]
        assert len(keys) == len(set(keys)), f"{task_type}: дублирующиеся ключи адаптеров"


def test_core_adapters_need_no_optional_dependency() -> None:
    #ядро обязано ставиться и работать только на sklearn (D-2)
    for adapter in (
        core.DummyClassifierAdapter(),
        core.LogisticRegressionAdapter(),
        core.RandomForestClassifierAdapter(),
        core.HistGradientBoostingClassifierAdapter(),
    ):
        assert adapter.availability().available
