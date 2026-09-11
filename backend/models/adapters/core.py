"""Ядро контендеров: только зависимости scikit-learn (D-2).

Baseline здесь не «ещё одна модель», а точка отсчёта. Без него метрика 0.88 ничего
не значит: на датасете с 88% одного класса это и есть результат постоянного ответа.
Поэтому baseline нельзя отключить, и он не участвует в борьбе за звание чемпиона.
"""
from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

from backend.models.base import Applicability, BaseAdapter, CostEstimate
from backend.models.context import DatasetContext, ResourceBudget

#выше этого числа объектов расстояние до каждого делает предсказание непригодным по времени
KNN_MAX_ROWS = 200_000
#в высокой размерности расстояния между точками сближаются и перестают различать объекты
KNN_MAX_DIMENSION = 50
#обучение SVM с ядром растёт как квадрат-куб числа объектов
SVM_MAX_ROWS = 20_000
#выше этого числа столбцов после кодирования плотная матрица становится основным расходом
WIDE_DIMENSION = 10_000
#датасет, на котором ансамбль деревьев считается десятки минут
LARGE_FOREST_ROWS = 1_000_000


class DummyClassifierAdapter(BaseAdapter):
    key = "baseline_majority"
    label = "Baseline: самый частый класс"
    family = "dummy"
    supported_tasks = frozenset({"binary", "multiclass"})
    preprocessing_profile = "passthrough_dummy"
    is_baseline = True

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:  # noqa: ARG002
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent")

    def estimate_cost(self, context: DatasetContext) -> CostEstimate:  # noqa: ARG002
        return CostEstimate(relative_units=0.0, note="Baseline не обучается на признаках.")


class DummyRegressorAdapter(BaseAdapter):
    key = "baseline_median"
    label = "Baseline: медиана"
    family = "dummy"
    supported_tasks = frozenset({"regression"})
    preprocessing_profile = "passthrough_dummy"
    supports_proba = False
    is_baseline = True

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:  # noqa: ARG002
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy="median")

    def estimate_cost(self, context: DatasetContext) -> CostEstimate:  # noqa: ARG002
        return CostEstimate(relative_units=0.0, note="Baseline не обучается на признаках.")


class LogisticRegressionAdapter(BaseAdapter):
    key = "logistic_regression"
    label = "Logistic Regression"
    family = "linear"
    supported_tasks = frozenset({"binary", "multiclass"})
    preprocessing_profile = "scaled_onehot"

    def applicability(self, context: DatasetContext) -> Applicability:
        if context.effective_dimension > 5 * max(1, context.n_train_rows):
            return Applicability(
                verdict="caution",
                reason=(
                    f"Признаков после кодирования ({context.effective_dimension}) кратно больше "
                    f"наблюдений ({context.n_train_rows}): решение будет определяться "
                    "регуляризацией сильнее, чем данными."
                ),
            )

        return Applicability(verdict="ok")

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        #балансировка классов сюда НЕ подставляется: она меняет калибровку вероятностей
        #и делает несравнимыми log loss и PR-AUC с моделями без неё. Это решение прогона,
        #одинаковое для всех контендеров или ни для кого (D-12)
        return {"max_iter": 1000}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=seed, n_jobs=budget.threads_per_fit, **params)


class RidgeAdapter(BaseAdapter):
    key = "ridge"
    label = "Ridge Regression"
    family = "linear"
    supported_tasks = frozenset({"regression"})
    preprocessing_profile = "scaled_onehot"
    supports_proba = False

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"alpha": 1.0}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:  # noqa: ARG002
        from sklearn.linear_model import Ridge

        return Ridge(random_state=seed, **params)


class _ForestBase(BaseAdapter):
    family = "tree_ensemble"
    preprocessing_profile = "tree_ordinal"

    def applicability(self, context: DatasetContext) -> Applicability:
        if context.n_train_rows > LARGE_FOREST_ROWS:
            return Applicability(
                verdict="caution",
                reason=(
                    f"{context.n_train_rows} обучающих строк: ансамбль деревьев здесь считается "
                    "десятки минут на фолд."
                ),
            )

        if context.estimated_matrix_bytes > 2 * 1024**3:
            return Applicability(
                verdict="caution",
                reason=(
                    f"Оценка матрицы признаков — {context.estimated_matrix_bytes / 1024**3:.1f} ГБ."
                ),
            )

        return Applicability(verdict="ok")

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"n_estimators": 200, "max_depth": 14, "min_samples_leaf": 2}

    def estimate_cost(self, context: DatasetContext) -> CostEstimate:
        return CostEstimate(relative_units=float(context.n_train_rows) / 100.0)


class RandomForestClassifierAdapter(_ForestBase):
    key = "random_forest"
    label = "Random Forest"
    supported_tasks = frozenset({"binary", "multiclass"})

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from sklearn.ensemble import RandomForestClassifier

        #n_jobs берётся из бюджета, а не -1: иначе несколько контендеров разом займут все ядра
        return RandomForestClassifier(random_state=seed, n_jobs=budget.threads_per_fit, **params)


class RandomForestRegressorAdapter(_ForestBase):
    key = "random_forest"
    label = "Random Forest"
    supported_tasks = frozenset({"regression"})
    supports_proba = False

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed, n_jobs=budget.threads_per_fit, **params)


class ExtraTreesClassifierAdapter(_ForestBase):
    key = "extra_trees"
    label = "Extra Trees"
    supported_tasks = frozenset({"binary", "multiclass"})

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(random_state=seed, n_jobs=budget.threads_per_fit, **params)


class ExtraTreesRegressorAdapter(_ForestBase):
    key = "extra_trees"
    label = "Extra Trees"
    supported_tasks = frozenset({"regression"})
    supports_proba = False

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(random_state=seed, n_jobs=budget.threads_per_fit, **params)


class _HistGradientBoostingBase(BaseAdapter):
    family = "gbdt"
    #модель работает с пропусками сама: подстановка медианы стёрла бы информацию о том,
    #что значения не было
    preprocessing_profile = "native_missing"

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"max_iter": 200, "learning_rate": 0.1}

    def estimate_cost(self, context: DatasetContext) -> CostEstimate:
        return CostEstimate(relative_units=float(context.n_train_rows) / 200.0)


class HistGradientBoostingClassifierAdapter(_HistGradientBoostingBase):
    key = "hist_gradient_boosting"
    label = "HistGradientBoosting"
    supported_tasks = frozenset({"binary", "multiclass"})

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:  # noqa: ARG002
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed, **params)


class HistGradientBoostingRegressorAdapter(_HistGradientBoostingBase):
    key = "hist_gradient_boosting"
    label = "HistGradientBoosting"
    supported_tasks = frozenset({"regression"})
    supports_proba = False

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:  # noqa: ARG002
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed, **params)


class _GuardedByScale(BaseAdapter):
    """Адаптеры, которые уместны только на данных подходящего масштаба."""

    def applicability(self, context: DatasetContext) -> Applicability:
        if context.effective_dimension > WIDE_DIMENSION:
            return Applicability(
                verdict="caution",
                reason=(
                    f"{context.effective_dimension} признаков после кодирования: плотная матрица "
                    "станет основным расходом памяти."
                ),
            )

        return Applicability(verdict="ok")


class SvmClassifierAdapter(_GuardedByScale):
    key = "svm_rbf"
    label = "SVM (RBF)"
    family = "svm"
    supported_tasks = frozenset({"binary", "multiclass"})
    preprocessing_profile = "scaled_onehot"

    def applicability(self, context: DatasetContext) -> Applicability:
        if context.n_train_rows > SVM_MAX_ROWS:
            return Applicability(
                verdict="skip",
                reason=(
                    f"{context.n_train_rows} обучающих строк при пределе {SVM_MAX_ROWS}: "
                    "время обучения растёт как квадрат-куб числа объектов, и контендер "
                    "занял бы больше времени, чем весь остальной турнир."
                ),
            )

        return super().applicability(context)

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        #вероятности нужны для ранговых метрик, но стоят внутренней кросс-валидации
        return {"kernel": "rbf", "probability": True}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:  # noqa: ARG002
        from sklearn.svm import SVC

        return SVC(random_state=seed, **params)

    def estimate_cost(self, context: DatasetContext) -> CostEstimate:
        return CostEstimate(
            relative_units=(float(context.n_train_rows) ** 2) / 1_000_000.0,
            note="Стоимость растёт квадратично.",
        )


class KnnClassifierAdapter(_GuardedByScale):
    key = "knn"
    label = "K Nearest Neighbors"
    family = "neighbors"
    supported_tasks = frozenset({"binary", "multiclass"})
    preprocessing_profile = "scaled_onehot"

    def applicability(self, context: DatasetContext) -> Applicability:
        if context.n_rows > KNN_MAX_ROWS:
            return Applicability(
                verdict="skip",
                reason=(
                    f"{context.n_rows} строк: расстояние считается до каждого объекта "
                    "на каждое предсказание, и inference окажется непригодным по времени."
                ),
            )

        if context.effective_dimension > KNN_MAX_DIMENSION:
            return Applicability(
                verdict="caution",
                reason=(
                    f"{context.effective_dimension} измерений: в высокой размерности расстояния "
                    "между объектами сближаются и перестают их различать."
                ),
            )

        return Applicability(verdict="ok")

    def default_params(self, context: DatasetContext) -> dict[str, Any]:  # noqa: ARG002
        return {"n_neighbors": 15}

    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator:  # noqa: ARG002
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_jobs=budget.threads_per_fit, **params)
