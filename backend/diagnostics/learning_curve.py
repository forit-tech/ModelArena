"""Кривая обучения: хватает ли данных.

Вопрос «нужно ли больше данных» обычно решают ощущением. Здесь он решается измерением:
модель обучается на растущих долях **той же обучающей части** и проверяется на **той же
проверочной**. Растёт качество с объёмом — данные ещё дают отдачу; вышло на полку —
следующая тысяча строк почти ничего не изменит.

Три границы, за которые слой не выходит.

**Разбиение берётся готовым.** Подвыборка режется внутри обучающей части фолда, проверочная
не трогается вовсе. Иначе кривая измеряла бы удачу разбиения, а не объём данных.

**Точек мало и они названы.** Два десятка точек не дают ничего сверх четырёх, зато стоят
в пять раз дороже. Стоимость считается заранее и возвращается вместе с результатом.

**Вывод не обещает будущего.** Из наблюдаемой кривой нельзя получить «добавьте 5000 строк
и получите +3%». Допустимые выводы: качество растёт, вышло на полку, разброс велик,
данных мало для устойчивого вывода.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from backend.arena.metrics import direction_of, evaluate, label_of, metric_keys, primary_metric
from backend.arena.predictions import SPLIT_CV
from backend.core.errors import ValidationError

DEFAULT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
#меньше этого числа строк в подвыборке обучение перестаёт быть осмысленным
MIN_SUBSET_ROWS = 20
#потолок на число обучений: дальше вычисление превращается во второй турнир
MAX_FITS = 40
#прирост в долях метрики, ниже которого рост считается остановившимся
PLATEAU_GAIN = 0.01


def estimate_cost(n_folds: int, fractions: tuple[float, ...] = DEFAULT_FRACTIONS) -> dict[str, Any]:
    fits = n_folds * len(fractions)
    return {
        "fits": fits,
        "folds": n_folds,
        "points": len(fractions),
        "within_limit": fits <= MAX_FITS,
        "limit": MAX_FITS,
        "note": (
            f"Потребуется {fits} обучений: {len(fractions)} размеров на {n_folds} фолдах. "
            "Модель обучается заново на каждой точке — иначе кривой не получить."
        ),
    }


def build_learning_curve(
    *,
    folds_path: Path,
    dataset_path: Path,
    task: dict[str, Any],
    contender: dict[str, Any],
    seed: int,
    threads: int = 1,
    metric: str | None = None,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    """Обучить участника на растущих подвыборках и измерить на тех же проверочных частях."""
    from sklearn.pipeline import Pipeline

    from backend.models.context import ResourceBudget
    from backend.models.registry import adapter_by_key
    from backend.preprocessing.profiles import (
        build_preprocessor,
        normalize_categoricals,
        split_feature_types,
    )
    from backend.tasks.spec import label_strings

    task_type = str(task.get("task_type", ""))
    chosen = metric or primary_metric(task_type)

    if chosen not in metric_keys(task_type):
        raise ValidationError(f"Метрика «{chosen}» не считается для задачи {task_type}.")

    splits = np.load(folds_path)
    n_folds = int(splits["n_folds"])
    cost = estimate_cost(n_folds, fractions)

    if not cost["within_limit"]:
        raise ValidationError(
            f"{cost['note']} Это больше предела в {MAX_FITS} обучений: уменьшите число "
            "фолдов или точек."
        )

    frame = pl.read_parquet(dataset_path)
    target_column = str(task.get("target_column", ""))
    usable = frame.filter(pl.col(target_column).is_not_null())
    feature_columns = list(task.get("feature_columns") or [])
    features = usable.select(feature_columns).to_pandas()
    types = split_feature_types(features)
    features = normalize_categoricals(features, types.categorical)

    target = (
        usable[target_column].cast(pl.Float64).to_numpy()
        if task_type == "regression"
        else np.asarray(label_strings(usable[target_column]).to_list(), dtype=object)
    )
    budget = ResourceBudget(max_parallel_fits=1, threads_per_fit=max(1, threads), memory_budget_mb=2048)
    adapter = adapter_by_key(task_type, str(contender.get("adapter_key")))
    class_labels = list(task.get("class_labels") or [])
    points: list[dict[str, Any]] = []
    generator = np.random.default_rng(seed)

    skipped: list[str] = []

    for fraction in fractions:
        scores: list[float] = []
        sizes: list[int] = []

        for fold in range(n_folds):
            train_index = splits[f"train_{fold}"]
            validation_index = splits[f"val_{fold}"]
            take = min(max(MIN_SUBSET_ROWS, round(len(train_index) * fraction)), len(train_index))

            if take < MIN_SUBSET_ROWS:
                skipped.append(
                    f"фолд {fold}, доля {fraction}: в обучающей части {len(train_index)} строк, "
                    f"меньше минимума {MIN_SUBSET_ROWS}"
                )
                continue

            #подвыборка режется ТОЛЬКО внутри обучающей части; проверочная неизменна,
            #иначе кривая измеряла бы разбиение, а не объём данных
            chosen_rows = (
                train_index
                if take == len(train_index)
                else np.sort(generator.choice(train_index, size=take, replace=False))
            )
            pipeline = Pipeline(
                [
                    ("preprocessing", build_preprocessor(contender["preprocessing_profile"], types)),
                    ("model", adapter.build(dict(contender.get("params") or {}), budget, seed)),
                ]
            )

            try:
                pipeline.fit(features.iloc[chosen_rows], target[chosen_rows])
                predicted = pipeline.predict(features.iloc[validation_index])
                #вероятности нужны ранговым метрикам: без них ROC-AUC не определён,
                #и кривая по умолчанию осталась бы пустой
                proba = _probabilities(pipeline, features.iloc[validation_index], class_labels)
            except Exception as error:  # noqa: BLE001
                #одна точка может не посчитаться — например, в подвыборке не оказалось
                #одного из классов. Это пропуск точки, а не провал кривой; но пропуск
                #обязан быть назван, иначе кривая молча окажется построенной по другим
                #данным, чем думает читающий
                skipped.append(
                    f"фолд {fold}, доля {fraction}: обучение не удалось "
                    f"({type(error).__name__})"
                )
                continue

            value = _score(
                predicted=predicted,
                proba=proba,
                truth=target[validation_index],
                task_type=task_type,
                class_labels=class_labels,
                positive_label=task.get("positive_label"),
                metric=chosen,
            )

            if value is not None:
                scores.append(value)
                sizes.append(len(chosen_rows))

        if scores:
            points.append(
                {
                    "fraction": fraction,
                    "train_rows": round(float(np.mean(sizes))),
                    "folds_measured": len(scores),
                    "mean": round(float(np.mean(scores)), 6),
                    "spread": round(float(np.std(scores, ddof=0)), 6) if len(scores) > 1 else None,
                    "per_fold": [round(value, 6) for value in scores],
                }
            )

    return {
        "available": bool(points),
        "metric": chosen,
        "metric_label": label_of(chosen),
        "higher_is_better": direction_of(chosen) == "higher",
        "cost": cost,
        "points": points,
        #пропущенные точки перечислены: кривая с дырой не должна выглядеть сплошной
        "skipped": skipped,
        "interpretation": _interpret(points, higher_is_better=direction_of(chosen) == "higher"),
    }


def _probabilities(pipeline: Any, part: Any, class_labels: list[str]) -> Any:
    """Вероятности в каноническом порядке меток либо None, если модель их не даёт."""
    if not class_labels:
        return None

    from backend.arena.predictions import align_probabilities

    try:
        raw = np.asarray(pipeline.predict_proba(part))
    except (AttributeError, NotImplementedError):
        return None

    model = pipeline.named_steps.get("model")
    classes = [str(value) for value in getattr(model, "classes_", [])]

    if not classes or raw.ndim != 2:
        return None

    aligned, _ = align_probabilities(raw, classes, class_labels)
    return aligned


def _score(
    *,
    predicted: Any,
    proba: Any,
    truth: Any,
    task_type: str,
    class_labels: list[str],
    positive_label: Any,
    metric: str,
) -> float | None:
    """Оценить точку тем же модулем метрик, что и весь остальной продукт."""
    if task_type == "regression":
        frame = pl.DataFrame(
            {
                "__row_id__": np.arange(len(truth), dtype=np.int64),
                "split": [SPLIT_CV] * len(truth),
                "fold": np.zeros(len(truth), dtype=np.int32),
                "y_true": np.asarray(truth, dtype=np.float64),
                "y_pred": np.asarray(predicted, dtype=np.float64),
            }
        )
    else:
        columns: dict[str, Any] = {
            "__row_id__": np.arange(len(truth), dtype=np.int64),
            "split": [SPLIT_CV] * len(truth),
            "fold": np.zeros(len(truth), dtype=np.int32),
            "y_true": [str(value) for value in truth],
            "y_pred": [str(value) for value in predicted],
        }

        if proba is not None:
            for position, label in enumerate(class_labels):
                columns[f"proba__{label}"] = np.asarray(proba[:, position], dtype=np.float32)

        frame = pl.DataFrame(columns)

    sets = evaluate(
        frame, task_type=task_type, class_labels=class_labels, positive_label=positive_label
    )

    if not sets:
        return None

    value = sets[0].by_key(metric)
    return value.value if value else None


def _interpret(points: list[dict[str, Any]], *, higher_is_better: bool) -> dict[str, Any]:
    """Вывод строго из наблюдаемой кривой, без обещаний про будущие строки."""
    if len(points) < 2:
        return {
            "verdict": "insufficient",
            "text": (
                "Точек слишком мало для вывода: кривая не построена. "
                "Это не значит, что данных достаточно или недостаточно."
            ),
        }

    values = [item["mean"] for item in points]
    first, last = values[0], values[-1]
    recent = (values[-1] - values[-2]) if higher_is_better else (values[-2] - values[-1])
    spreads = [item["spread"] for item in points if item["spread"] is not None]
    mean_spread = float(np.mean(spreads)) if spreads else 0.0

    if mean_spread and abs(recent) < mean_spread:
        return {
            "verdict": "noisy",
            "text": (
                f"Прирост на последнем шаге ({recent:+.4f}) меньше разброса между фолдами "
                f"({mean_spread:.4f}). По этой кривой нельзя сказать, продолжает ли качество "
                "расти: различие тонет в шуме."
            ),
        }

    if recent >= PLATEAU_GAIN:
        return {
            "verdict": "growing",
            "text": (
                f"Качество продолжает расти: от {first:.4f} до {last:.4f}, прирост на последнем "
                f"шаге {recent:+.4f}. На этих данных дополнительные строки ещё дают отдачу. "
                "Насколько именно — из кривой не следует."
            ),
        }

    return {
        "verdict": "plateau",
        "text": (
            f"Кривая вышла на полку: от {first:.4f} до {last:.4f}, прирост на последнем шаге "
            f"{recent:+.4f} при пороге {PLATEAU_GAIN}. Увеличение объёма данных того же рода "
            "заметного улучшения не даст — узкое место в другом: в признаках или в модели."
        ),
    }
