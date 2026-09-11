"""Признак, который в одиночку почти решает задачу, и пропуск, кодирующий цель.

Обе проверки измеряются **вне обучающей части**: модель для проверки обучается только
на train-части фолда и оценивается на валидационной. Инструмент, ищущий утечку,
не имеет права её создавать (D-23), поэтому ничего не обучается на полном датасете.

Уровень доказательности — `evidence`, а не `structural`: очень сильный признак бывает
и честным. Формулировка это признаёт прямо, и вывод оставляет человеку.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from backend.leakage.report import (
    MIN_ROWS_FOR_MISSINGNESS,
    MISSINGNESS_ASSOCIATION,
    SINGLE_FEATURE_CLASSIFICATION,
    SINGLE_FEATURE_REGRESSION,
    LeakageSignal,
    NotEvaluated,
)
from backend.protocol.folds import FoldPlan
from backend.tasks.spec import TaskSpec, label_strings

#бюджет обучений неглубокого дерева за один прогон. Ограничение необходимо: без него
#широкий датасет даёт число обучений, растущее как признаки × фолды. Но если бюджет
#исчерпан, непроверенные признаки обязаны быть НАЗВАНЫ, а не пропущены молча (D-23)
PROBE_BUDGET = 400
MIN_ROWS_FOR_PROBE = 40


def check(usable: pl.DataFrame, spec: TaskSpec, folds: FoldPlan) -> list[LeakageSignal]:
    signals = _single_feature(usable, spec, folds)
    signals.extend(_linear_reconstruction(usable, spec, folds))
    signals.extend(_missingness(usable, spec))
    return signals


def _linear_reconstruction(
    usable: pl.DataFrame, spec: TaskSpec, folds: FoldPlan
) -> list[LeakageSignal]:
    """Почти-копия цели в регрессии.

    Неглубокое дерево здесь плохой детектор: у него шестнадцать листьев, и цель
    с сотней различных значений оно не воспроизведёт, даже если признак — это она же
    плюс микроскопический шум. Почти-копия по построению линейна, поэтому и проверяется
    прямой, подогнанной **только на обучающей части** фолда.
    """
    if spec.task_type != "regression" or not folds.validation_folds:
        return []

    target = usable[spec.target_column].cast(pl.Float64).to_numpy()

    if not np.isfinite(target).all() or float(np.std(target)) == 0:
        return []

    from sklearn.metrics import r2_score

    suspicious: list[dict[str, object]] = []

    for name in spec.feature_columns:
        if name not in usable.columns or not usable[name].dtype.is_numeric():
            continue

        values = usable[name].cast(pl.Float64).to_numpy()

        if not np.isfinite(values).all() or float(np.std(values)) == 0:
            continue

        scores: list[float] = []

        for index, validation in enumerate(folds.validation_folds):
            training = folds.training_for(index)

            if len(training) < 10 or len(validation) < 5:
                continue

            slope, intercept = np.polyfit(values[training], target[training], 1)
            predicted = slope * values[validation] + intercept
            scores.append(float(r2_score(target[validation], predicted)))

        if scores and float(np.mean(scores)) >= SINGLE_FEATURE_REGRESSION:
            suspicious.append({"column": name, "score": round(float(np.mean(scores)), 6)})

    if not suspicious:
        return []

    columns = [str(item["column"]) for item in suspicious]

    return [
        LeakageSignal(
            code="single_feature_reconstructs_target",
            evidence_level="evidence",
            scope="features",
            columns=columns,
            observed_value=suspicious,
            threshold=f"{SINGLE_FEATURE_REGRESSION} R² линейной подгонки вне обучающей части",
            explanation=(
                "Прямая, подогнанная только на обучающей части, восстанавливает цель из "
                + ", ".join(f"«{item['column']}» (R² {item['score']:.4f})" for item in suspicious[:3])
                + " на данных, которых не видела."
            ),
            mechanism=(
                "Почти линейная связь такой точности означает, что колонка пересчитана из цели "
                "или измеряет ровно ту же величину. Все модели получат R² около единицы, "
                "и сравнивать окажется нечего. Точного восстановления здесь нет, поэтому "
                "структурной проверкой это не ловится — вывод остаётся за человеком."
            ),
            suggested_action=(
                "Проверьте происхождение колонки: вычислена ли она из цели или измерена "
                "независимо."
            ),
        )
    ]


def _encode(series: pl.Series) -> np.ndarray | None:
    if series.dtype.is_numeric() and series.dtype != pl.Boolean:
        return series.cast(pl.Float64).fill_null(-999_999.0).to_numpy().reshape(-1, 1)

    if series.dtype.is_temporal():
        return series.cast(pl.Int64).fill_null(0).to_numpy().astype(float).reshape(-1, 1)

    #категории кодируются порядковыми номерами: для одного неглубокого дерева этого хватает,
    #а порядок номеров на результат не влияет, потому что дерево делает пороговые разбиения
    codes = series.cast(pl.String).fill_null("__null__").cast(pl.Categorical).to_physical()
    array = codes.to_numpy().astype(float)

    return array.reshape(-1, 1) if len(set(array.tolist())) >= 2 else None


def _single_feature(usable: pl.DataFrame, spec: TaskSpec, folds: FoldPlan) -> list:
    if usable.height < MIN_ROWS_FOR_PROBE or not folds.validation_folds:
        return [
            NotEvaluated(
                check="single_feature_reconstructs_target",
                scope="features",
                reason=(
                    f"Для измерения нужно минимум {MIN_ROWS_FOR_PROBE} строк с известной целью "
                    f"и хотя бы один фолд; есть {usable.height} строк "
                    f"и {len(folds.validation_folds)} фолдов."
                ),
            )
        ]

    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    is_regression = spec.task_type == "regression"
    target = (
        usable[spec.target_column].cast(pl.Float64).to_numpy()
        if is_regression
        else label_strings(usable[spec.target_column]).to_numpy()
    )

    if is_regression and (not np.isfinite(target).all() or float(np.std(target)) == 0):
        return [
            NotEvaluated(
                check="single_feature_reconstructs_target",
                scope="target",
                reason="Цель содержит бесконечности или не имеет разброса — измерять нечего.",
            )
        ]

    suspicious: list[dict[str, object]] = []
    skipped: list[NotEvaluated] = []
    budget = PROBE_BUDGET
    checked: list[str] = []

    for name in spec.feature_columns:
        if name not in usable.columns:
            continue

        if budget <= 0:
            skipped.append(
                NotEvaluated(
                    check="single_feature_reconstructs_target",
                    scope=f"column:{name}",
                    reason=(
                        f"Исчерпан бюджет измерений ({PROBE_BUDGET} обучений). Проверено "
                        f"{len(checked)} признаков, остальные не измерялись."
                    ),
                )
            )
            continue

        encoded = _encode(usable[name])

        if encoded is None:
            skipped.append(
                NotEvaluated(
                    check="single_feature_reconstructs_target",
                    scope=f"column:{name}",
                    reason=(
                        f"Тип {usable[name].dtype} не кодируется для пробного дерева "
                        "или в колонке одно значение."
                    ),
                )
            )
            continue

        scores: list[float] = []

        for index, validation in enumerate(folds.validation_folds):
            training = folds.training_for(index)

            if len(training) < 10 or len(validation) < 5:
                continue

            budget -= 1

            try:
                if is_regression:
                    model = DecisionTreeRegressor(max_depth=4, random_state=0)
                    model.fit(encoded[training], target[training])
                    scores.append(float(r2_score(target[validation], model.predict(encoded[validation]))))
                else:
                    if len(set(target[training].tolist())) < 2:
                        continue

                    model = DecisionTreeClassifier(max_depth=4, random_state=0)
                    model.fit(encoded[training], target[training])
                    scores.append(
                        float(
                            balanced_accuracy_score(
                                target[validation], model.predict(encoded[validation])
                            )
                        )
                    )
            except ValueError:
                #отдельный признак может не подойти дереву на конкретном фолде;
                #это не ошибка отчёта, но и не повод считать проверку выполненной
                continue

        if not scores:
            skipped.append(
                NotEvaluated(
                    check="single_feature_reconstructs_target",
                    scope=f"column:{name}",
                    reason="Ни на одном фолде пробное дерево обучить не удалось.",
                )
            )
            continue

        checked.append(name)
        mean_score = float(np.mean(scores))
        threshold = SINGLE_FEATURE_REGRESSION if is_regression else SINGLE_FEATURE_CLASSIFICATION

        if mean_score >= threshold:
            suspicious.append({"column": name, "score": round(mean_score, 6)})

    if not suspicious:
        return list(skipped)

    metric = "R²" if is_regression else "Balanced Accuracy"
    columns = [str(item["column"]) for item in suspicious]

    return [
        LeakageSignal(
            code="single_feature_reconstructs_target",
            evidence_level="evidence",
            scope="features",
            columns=columns,
            observed_value=suspicious,
            threshold=(
                f"{SINGLE_FEATURE_REGRESSION if is_regression else SINGLE_FEATURE_CLASSIFICATION} "
                f"{metric} вне обучающей части"
            ),
            explanation=(
                "Неглубокое дерево, обученное только на "
                + ", ".join(f"«{item['column']}» ({metric} {item['score']:.4f})" for item in suspicious[:3])
                + ", почти полностью восстанавливает цель на данных, которых не видело."
            ),
            mechanism=(
                "Если колонка становится известна только после наступления события, все модели "
                "получат почти идеальную метрику, и сравнение потеряет смысл. Но такой результат "
                "бывает и у честного признака — например, у прямого измерения того же процесса. "
                "Отличить одно от другого можно только знанием предметной области."
            ),
            suggested_action=(
                "Проверьте, доступна ли эта колонка в момент предсказания. Если заполняется "
                "после — исключите её; если известна заранее, это просто сильный признак."
            ),
        ),
        *skipped,
    ]


def _missingness(usable: pl.DataFrame, spec: TaskSpec) -> list[LeakageSignal]:
    """Сам факт пропуска может кодировать цель.

    Классический механизм: поле заполняется только для одного исхода. Значение при этом
    выглядит безобидно, а `is_null` несёт ответ.
    """
    if spec.task_type == "regression":
        return [
            NotEvaluated(
                check="missingness_encodes_target",
                scope="features",
                reason=(
                    "Механизм проверяется через долю положительного класса, которой "
                    "у регрессии нет. Связь пропуска с числовой целью не измерялась."
                ),
            )
        ]

    if usable.height < MIN_ROWS_FOR_MISSINGNESS * 2:
        return [
            NotEvaluated(
                check="missingness_encodes_target",
                scope="features",
                reason=(
                    f"Нужно минимум {MIN_ROWS_FOR_MISSINGNESS * 2} строк, чтобы разница долей "
                    f"что-то значила; есть {usable.height}."
                ),
            )
        ]

    target_text = label_strings(usable[spec.target_column])
    positive = spec.positive_label or (spec.class_labels[0] if spec.class_labels else None)

    if positive is None:
        return []

    is_positive = (target_text == positive).to_numpy()
    suspicious: list[dict[str, object]] = []

    for name in spec.feature_columns:
        if name not in usable.columns:
            continue

        missing = usable[name].is_null().to_numpy()
        present_count = int((~missing).sum())
        missing_count = int(missing.sum())

        if missing_count < MIN_ROWS_FOR_MISSINGNESS or present_count < MIN_ROWS_FOR_MISSINGNESS:
            continue

        rate_missing = float(is_positive[missing].mean())
        rate_present = float(is_positive[~missing].mean())
        difference = abs(rate_missing - rate_present)

        if difference >= MISSINGNESS_ASSOCIATION:
            suspicious.append(
                {
                    "column": name,
                    "positive_rate_when_missing": round(rate_missing, 4),
                    "positive_rate_when_present": round(rate_present, 4),
                }
            )

    if not suspicious:
        return []

    columns = [str(item["column"]) for item in suspicious]

    return [
        LeakageSignal(
            code="missingness_encodes_target",
            evidence_level="evidence",
            scope="features",
            columns=columns,
            observed_value=suspicious,
            threshold=f"разница долей положительного класса {MISSINGNESS_ASSOCIATION:.0%}",
            explanation=(
                "Сам факт пропуска в "
                + ", ".join(
                    f"«{item['column']}» ({item['positive_rate_when_missing']:.0%} против "
                    f"{item['positive_rate_when_present']:.0%})"
                    for item in suspicious[:3]
                )
                + " сильно связан с целью."
            ),
            mechanism=(
                "Пропуск переживает любую подстановку: заполнив медианой, вы сохраните признак "
                "«здесь было пусто», и модель выучит именно его. Если поле заполняется только "
                "для одного исхода, это ответ, замаскированный под отсутствие данных."
            ),
            suggested_action=(
                "Выясните, почему поле пусто именно у этих строк. Если пропуск возникает "
                "после наступления события — исключите колонку."
            ),
        )
    ]
