"""Профили препроцессинга под семейства моделей.

Честность сравнения задаётся протоколом, а не одинаковым кодированием (D-7): одинаковые
фолды, seed, набор признаков и метрики — общие для всех, а способ подготовки признаков
может различаться. Гнать градиентный бустинг через `StandardScaler` и one-hot не «честно»,
а искусственно ухудшать одного участника.

**Весь препроцессинг живёт внутри `Pipeline`.** Это не соглашение, а структурная
гарантия: функции, которая обучила бы трансформер на полном датасете до разбиения,
здесь просто нет. Статистики считаются по обучающей части фолда и применяются
к проверочной — иначе проверка видела бы собственное среднее.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ProfileKey = Literal["scaled_onehot", "tree_ordinal", "native_missing", "passthrough_dummy"]

#one-hot по категории с сотнями значений взрывает размерность, поэтому редкие уровни
#собираются в общий; предел совпадает с тем, что учитывает Readiness при оценке размерности
MAX_CATEGORIES = 50
MIN_CATEGORY_FREQUENCY = 2
#значение для неизвестной категории в порядковом кодировании: -1 не встречается среди кодов
UNKNOWN_ORDINAL = -1
DATETIME_PARTS = ("year", "month", "day", "dayofweek", "quarter", "epoch_days")


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Календарные части вместо самой даты.

    Без этого дата попадает в one-hot и порождает по бинарному признаку на каждую
    конкретную дату: такие признаки не переносятся на новые данные и раздувают модель.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> DateTimeFeatures:  # noqa: ARG002, N803
        #статистик обучения нет: календарные части считаются построчно.
        #fit нужен только для интерфейса sklearn и фиксации имён колонок
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        parts: list[np.ndarray] = []

        for name in self.feature_names_in_:
            values = _coerce_datetime(X[name])
            parts.append(values.dt.year.to_numpy(dtype=float))
            parts.append(values.dt.month.to_numpy(dtype=float))
            parts.append(values.dt.day.to_numpy(dtype=float))
            parts.append(values.dt.dayofweek.to_numpy(dtype=float))
            parts.append(values.dt.quarter.to_numpy(dtype=float))
            parts.append((values.astype("int64") / (86_400 * 1_000_000_000)).to_numpy(dtype=float))

        if not parts:
            return np.empty((len(X), 0), dtype=float)

        matrix = np.column_stack(parts)
        #astype("int64") превращает NaT в минимальное int64 — пропуски восстанавливаются явно,
        #иначе «нет даты» стало бы датой из 1677 года и выглядело бы как настоящее значение
        return np.where(np.isfinite(matrix) & (matrix > -1e17), matrix, np.nan)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:  # noqa: ARG002
        #понятные имена: важность читается как signup_date__month, а не как column_17
        return np.array(
            [f"{name}__{part}" for name in self.feature_names_in_ for part in DATETIME_PARTS]
        )


def _coerce_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    return pd.to_datetime(series, errors="coerce", format="mixed")


@dataclass(frozen=True)
class FeatureTypes:
    numeric: list[str]
    datetime: list[str]
    categorical: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "datetime": self.datetime, "categorical": self.categorical}


def split_feature_types(features: pd.DataFrame) -> FeatureTypes:
    """Разделить признаки по фактическому типу, а не по имени колонки."""
    numeric: list[str] = []
    datetime: list[str] = []
    categorical: list[str] = []

    for name in features.columns:
        series = features[name]

        if pd.api.types.is_bool_dtype(series):
            #булев признак — это категория из двух значений, а не число
            categorical.append(name)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(name)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime.append(name)
        else:
            categorical.append(name)

    return FeatureTypes(numeric=numeric, datetime=datetime, categorical=categorical)


def normalize_categoricals(features: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Привести категории к строкам, сохранив пропуски как `np.nan`.

    Одинаковое строковое представление нужно, чтобы `True` и `"True"` из разных источников
    не стали разными категориями.

    Пропуск записывается **именно `np.nan`**, а не `None`. Причина не стилистическая:
    в колонке типа `object` scikit-learn ищет пропуски проверкой `X != X`, а `None != None`
    равно `False`. Значение `None` пропуском не признаётся, проходит мимо импьютера
    и становится полноправной категорией — то есть «значения нет» превращается в значение.

    Хуже того, результат зависел от версии pandas: одни версии сохраняли `None` после `map`,
    другие приводили его к `NaN`. Один и тот же снимок давал **разные матрицы признаков**
    в разных окружениях, а значит и разные метрики, — при том что воспроизводимость
    эксперимента заявлена свойством системы. Обнаружено расхождением CI и рабочей машины.
    """
    if not categorical:
        return features

    normalized = features.copy()

    for name in categorical:
        #astype(object) фиксирует тип колонки: без него колонка целиком из пропусков
        #получила бы float64 и перестала быть категориальной посреди конвейера
        normalized[name] = (
            normalized[name]
            .map(lambda value: np.nan if pd.isna(value) else str(value))
            .astype(object)
        )

    return normalized


def build_preprocessor(profile: ProfileKey, types: FeatureTypes) -> ColumnTransformer:
    """Собрать препроцессор под выбранный профиль."""
    if profile == "passthrough_dummy":
        #baseline не смотрит на признаки вовсе: любая подготовка была бы декорацией
        return ColumnTransformer(transformers=[], remainder="drop")

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if types.numeric:
        transformers.append(("numeric", _numeric_pipeline(profile), types.numeric))

    if types.datetime:
        transformers.append(("datetime", _datetime_pipeline(profile), types.datetime))

    if types.categorical:
        transformers.append(("categorical", _categorical_pipeline(profile), types.categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _numeric_pipeline(profile: ProfileKey) -> Pipeline:
    if profile == "native_missing":
        #модель работает с пропусками сама; подстановка медианы лишила бы её информации
        return Pipeline([("identity", "passthrough")])

    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]

    if profile == "scaled_onehot":
        #масштаб важен только там, где расстояние или регуляризация зависят от единиц измерения
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def _datetime_pipeline(profile: ProfileKey) -> Pipeline:
    steps: list[tuple[str, Any]] = [("calendar", DateTimeFeatures())]

    if profile != "native_missing":
        steps.append(("imputer", SimpleImputer(strategy="median")))

    if profile == "scaled_onehot":
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def _categorical_pipeline(profile: ProfileKey) -> Pipeline:
    if profile == "scaled_onehot":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=MIN_CATEGORY_FREQUENCY,
                        max_categories=MAX_CATEGORIES,
                        sparse_output=False,
                    ),
                ),
            ]
        )

    #деревьям порядковый код не мешает: разбиение идёт по порогу, а не по величине.
    #Неизвестная категория получает значение вне диапазона кодов, а не «самую частую»
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=UNKNOWN_ORDINAL,
                    encoded_missing_value=UNKNOWN_ORDINAL,
                ),
            ),
        ]
    )
