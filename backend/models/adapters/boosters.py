"""Бустеры — опциональные участники (D-2).

Ядро ставится только на зависимостях scikit-learn, а эти три подключаются через
`pip install modelarena[boosters]`. Ключевое: **недоступный адаптер не исчезает**.
Он остаётся в списке контендеров со статусом и текстом, что именно поставить.

Пустое место в leaderboard читается как «модель проиграла», а не как «её не запускали»,
и это ровно та молчаливая неправда, которой в инструменте про честное сравнение быть
не должно.
"""
from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

from backend.models.base import BaseAdapter, CostEstimate
from backend.models.context import DatasetContext, ResourceBudget


class _BoosterBase(BaseAdapter):
    family = "gbdt"
    preprocessing_profile = "native_missing"

    def estimate_cost(self, context: DatasetContext) -> CostEstimate:
        return CostEstimate(relative_units=float(context.n_train_rows) / 200.0)


class XgboostClassifierAdapter(_BoosterBase):
    key = "xgboost"
    label = "XGBoost"
    supported_tasks = frozenset({"binary", "multiclass"})
    requires_module = "xgboost"
    install_hint = "Установите: pip install modelarena[boosters]."

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=seed, n_jobs=budget.threads_per_fit, **params)


class XgboostRegressorAdapter(_BoosterBase):
    key = "xgboost"
    label = "XGBoost"
    supported_tasks = frozenset({"regression"})
    supports_proba = False
    requires_module = "xgboost"
    install_hint = "Установите: pip install modelarena[boosters]."

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed, n_jobs=budget.threads_per_fit, **params)


class LightgbmClassifierAdapter(_BoosterBase):
    key = "lightgbm"
    label = "LightGBM"
    supported_tasks = frozenset({"binary", "multiclass"})
    requires_module = "lightgbm"
    install_hint = "Установите: pip install modelarena[boosters]."

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"n_estimators": 300, "learning_rate": 0.1}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=seed, n_jobs=budget.threads_per_fit, verbose=-1, **params
        )


class LightgbmRegressorAdapter(_BoosterBase):
    key = "lightgbm"
    label = "LightGBM"
    supported_tasks = frozenset({"regression"})
    supports_proba = False
    requires_module = "lightgbm"
    install_hint = "Установите: pip install modelarena[boosters]."

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"n_estimators": 300, "learning_rate": 0.1}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=budget.threads_per_fit, verbose=-1, **params)


class CatboostClassifierAdapter(_BoosterBase):
    key = "catboost"
    label = "CatBoost"
    supported_tasks = frozenset({"binary", "multiclass"})
    requires_module = "catboost"
    install_hint = "Установите: pip install modelarena[boosters]."

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"iterations": 300, "learning_rate": 0.1, "depth": 6}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            random_seed=seed, thread_count=budget.threads_per_fit, verbose=False, **params
        )


class CatboostRegressorAdapter(_BoosterBase):
    key = "catboost"
    label = "CatBoost"
    supported_tasks = frozenset({"regression"})
    supports_proba = False
    requires_module = "catboost"
    install_hint = "Установите: pip install modelarena[boosters]."

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"iterations": 300, "learning_rate": 0.1, "depth": 6}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            random_seed=seed, thread_count=budget.threads_per_fit, verbose=False, **params
        )
