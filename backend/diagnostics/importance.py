"""Влияние признаков: перестановочная важность без причинных обещаний.

Что здесь измеряется: насколько ухудшится метрика, если значения одного признака
перемешать между строками. Это утверждение о **модели и данных**, а не о мире.

Чего здесь нет и не будет: фразы «признак X влияет на целевую переменную». Перемешивание
показывает, на что опирается конкретная обученная модель, — а она могла опереться
на следствие вместо причины, на посредника или на артефакт сбора данных. Разница между
«модель использует X» и «X определяет Y» — это разница между наблюдением и выводом,
которого из этого измерения не следует.

Измеряется на **holdout** — части, которую сохранённая модель не видела ни разу. Мерить
на обучающих строках бессмысленно: модель их запомнила, и перемешивание покажет силу
запоминания, а не переносимость.

Модель при этом **не переобучается ни разу**: перестановка меняет вход, а не модель.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from backend.arena.metrics import direction_of, label_of, metric_keys, primary_metric
from backend.arena.predictions import ROW_ID, SPLIT_HOLDOUT
from backend.core.errors import ValidationError

#по умолчанию: достаточно, чтобы увидеть разброс, и дёшево по времени
DEFAULT_REPEATS = 5
MAX_REPEATS = 20
#меньше этого числа строк перестановка даёт шум вместо измерения
MIN_ROWS = 30
#корреляция, выше которой важность между признаками размазывается
HIGH_CORRELATION = 0.8


@dataclass
class ImportanceRow:
    feature: str
    mean_drop: float
    std_drop: float
    per_repeat: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "mean_drop": self.mean_drop,
            "std_drop": self.std_drop,
            "per_repeat": self.per_repeat,
        }


def permutation_importance_for(
    pipeline: Any,
    manifest: dict[str, Any],
    snapshot: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    metric: str | None = None,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
) -> dict[str, Any]:
    """Посчитать перестановочную важность на holdout.

    `snapshot` читается с `__row_id__`, `predictions` даёт, какие строки были holdout
    и каков там факт. Так измерение опирается ровно на те строки, которых сохранённая
    модель не видела.
    """
    from sklearn.inspection import permutation_importance

    task = dict(manifest.get("task") or {})
    task_type = str(task.get("task_type", ""))
    chosen = metric or primary_metric(task_type)

    if chosen not in metric_keys(task_type):
        raise ValidationError(
            f"Метрика «{chosen}» не считается для задачи {task_type}."
        )

    holdout = predictions.filter(pl.col("split") == SPLIT_HOLDOUT)

    if holdout.height < MIN_ROWS:
        return {
            "available": False,
            "reason": (
                f"В holdout {holdout.height} строк — меньше {MIN_ROWS}. Перестановка на таком "
                "объёме даёт шум вместо измерения, и показывать его как важность нельзя."
            ),
        }

    schema = dict(manifest.get("feature_schema") or {})
    columns = list(schema.get("columns") or [])
    identifiers = holdout[ROW_ID].to_list()
    rows = snapshot.filter(pl.col(ROW_ID).is_in(identifiers)).sort(ROW_ID)
    ordered_holdout = holdout.sort(ROW_ID)

    if rows.height != ordered_holdout.height:
        raise ValidationError(
            "Строки holdout не сопоставились со снимком датасета: измерять важность не на чем."
        )

    from backend.preprocessing.profiles import normalize_categoricals

    features = normalize_categoricals(
        rows.select(columns).to_pandas(), list(schema.get("categorical") or [])
    )
    truth = _target_values(ordered_holdout, task_type)
    repeats = max(1, min(int(repeats), MAX_REPEATS))

    result = permutation_importance(
        pipeline,
        features,
        truth,
        scoring=_scorer(chosen, task_type, task.get("positive_label"), task.get("class_labels")),
        n_repeats=repeats,
        #случайность управляемая: повторный запрос на тех же данных даёт тот же ответ
        random_state=seed,
        n_jobs=1,
    )

    ranked = [
        ImportanceRow(
            feature=name,
            mean_drop=round(float(result.importances_mean[position]), 6),
            std_drop=round(float(result.importances_std[position]), 6),
            per_repeat=[round(float(value), 6) for value in result.importances[position]],
        )
        for position, name in enumerate(columns)
    ]
    ranked.sort(key=lambda item: item.mean_drop, reverse=True)

    return {
        "available": True,
        "metric": chosen,
        "metric_label": label_of(chosen),
        "higher_is_better": direction_of(chosen) == "higher",
        "measured_on": "holdout",
        "n_rows": ordered_holdout.height,
        "repeats": repeats,
        "seed": seed,
        "rows": [item.to_dict() for item in ranked],
        "correlated_groups": _correlated(rows, schema),
        "interpretation": (
            "Число показывает, на сколько падает "
            f"{label_of(chosen)}, если значения признака перемешать между строками. "
            "Это измерение того, на что опирается обученная модель, а не утверждение "
            "о влиянии признака на целевую переменную."
        ),
        "caveats": _caveats(ranked),
    }


def _target_values(holdout: pl.DataFrame, task_type: str) -> np.ndarray:
    if task_type == "regression":
        return holdout["y_true"].to_numpy().astype(np.float64)

    return np.asarray([str(value) for value in holdout["y_true"].to_list()], dtype=object)


def _scorer(metric: str, task_type: str, positive_label: Any, class_labels: Any) -> Any:
    """Оценщик, соответствующий выбранной метрике.

    Берётся из sklearn, а не пишется заново: своя реализация метрики рядом с общим
    модулем метрик рано или поздно разойдётся с ним в мелочах, и важность окажется
    посчитанной по чуть другой величине.
    """
    from sklearn.metrics import get_scorer, make_scorer, roc_auc_score

    if task_type == "regression":
        return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
                "r2": "r2"}.get(metric, "neg_root_mean_squared_error")

    labels = [str(item) for item in (class_labels or [])]

    if metric == "roc_auc" and len(labels) == 2 and positive_label:
        return make_scorer(
            roc_auc_score,
            response_method="predict_proba",
            #положительный класс берётся из постановки задачи, а не угадывается порядком
            labels=labels,
            multi_class="raise",
        )

    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1_weighted" if len(labels) > 2 else "f1",
        "macro_f1": "f1_macro",
        "precision": "precision_weighted" if len(labels) > 2 else "precision",
        "recall": "recall_weighted" if len(labels) > 2 else "recall",
    }.get(metric, get_scorer("accuracy"))


def _correlated(rows: pl.DataFrame, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Пары сильно связанных числовых признаков.

    Важность между такими признаками размазывается: модель может опереться на один
    и почти не тронуть второй, хотя содержательно они несут одно и то же. Без этого
    предупреждения нулевая важность читается как «признак не нужен».
    """
    numeric = [name for name in (schema.get("numeric") or []) if name in rows.columns]

    if len(numeric) < 2:
        return []

    matrix = rows.select(numeric).to_pandas().corr(numeric_only=True).to_numpy()
    pairs: list[dict[str, Any]] = []

    for first in range(len(numeric)):
        for second in range(first + 1, len(numeric)):
            value = matrix[first][second]

            if np.isfinite(value) and abs(value) >= HIGH_CORRELATION:
                pairs.append(
                    {
                        "features": [numeric[first], numeric[second]],
                        "correlation": round(float(value), 4),
                    }
                )

    return pairs


def _caveats(ranked: list[ImportanceRow]) -> list[str]:
    notes = [
        "Перестановочная важность измеряет опору конкретной обученной модели, "
        "а не причинную связь: модель могла опереться на следствие или на посредника.",
    ]
    unstable = [item for item in ranked if item.std_drop > abs(item.mean_drop)]

    if unstable:
        notes.append(
            "У признаков "
            + ", ".join(f"«{item.feature}»" for item in unstable[:5])
            + " разброс между повторами превышает саму величину: их важность измерена "
            "неустойчиво, и порядок между ними случаен."
        )

    negative = [item for item in ranked if item.mean_drop < 0]

    if negative:
        notes.append(
            "Отрицательная важность означает, что после перемешивания метрика стала лучше. "
            "Это признак шума, а не вредного признака: на таком объёме различие "
            "не отличается от случайного."
        )

    return notes
