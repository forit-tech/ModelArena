"""Обучение одного контендера в отдельном процессе.

Почему процесс, а не поток. Нативная библиотека умеет три вещи, от которых поток
не защищает: зависнуть внутри C++, где Python не получит управление и не проверит
никакой флаг отмены; выесть память так, что упадёт весь backend; и упасть по segfault,
унеся с собой FastAPI вместе с уже посчитанными результатами. Отдельный процесс делает
кнопку «Остановить» настоящей — его можно завершить, — и превращает падение стороннего
кода в отказ **одного** участника.

**Модуль на уровне импорта тянет только стандартную библиотеку.** Это не стиль:
при способе запуска `spawn` дочерний процесс импортирует этот модуль, чтобы найти
функцию, и любой numpy или sklearn здесь загрузил бы BLAS раньше, чем мы успеем
ограничить число его потоков. Тогда два контендера по семь потоков дали бы не
четырнадцать рабочих потоков, а двойной захват всех ядер.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

#переменные, которыми ограничивается внутренний параллелизм численных библиотек.
#Ставятся до первого импорта numpy: библиотеки читают их при загрузке
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
#сколько ждать, пока последние события уйдут родителю, прежде чем завершать процесс
EVENT_FLUSH_SECONDS = 5.0
RESULT_FILE = "result.json"
PREDICTIONS_FILE = "predictions.parquet"
TRACEBACK_FILE = "traceback.log"


class ContenderError(Exception):
    """Отказ контендера с машинным кодом. Полный стек остаётся на сервере."""

    def __init__(self, code: str, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or traceback.format_exc()


def apply_thread_limits(threads: int) -> None:
    value = str(max(1, int(threads)))

    for name in THREAD_VARIABLES:
        os.environ[name] = value


def run_contender(payload: dict[str, Any], events: Any) -> None:
    """Точка входа дочернего процесса.

    Наружу отсюда не выходит ни одного исключения: аварийное завершение процесса
    родитель обязан отличать от обработанной ошибки контендера, и для этого ошибка
    должна дойти до него как записанный результат, а не как код возврата.
    """
    apply_thread_limits(payload.get("threads_per_fit", 1))
    staging = Path(payload["staging_directory"])
    key = payload["contender_key"]
    started = time.monotonic()

    try:
        result = _train(payload, events, staging)
    except ContenderError as failure:
        _write_traceback(staging, failure.detail)
        result = _failure_result(key, failure.code, failure.message, started)
    except BaseException as error:  # noqa: BLE001
        #сюда попадает всё непредусмотренное, включая MemoryError. Полный стек уходит
        #в файл прогона, наружу — код и одна строка: в сообщении контендера не должно
        #быть путей и внутренностей окружения
        _write_traceback(staging, traceback.format_exc())
        result = _failure_result(
            key, "training_failed", f"Непредвиденная ошибка: {type(error).__name__}.", started
        )

    try:
        (staging / RESULT_FILE).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    except OSError as error:
        #результат некуда записать: родитель увидит процесс, завершившийся без результата,
        #и назовёт это аварией — что и произошло
        _emit(events, {"kind": "worker_failed", "contender_key": key, "message": str(error)})
        return

    _emit(events, {"kind": "worker_finished", "contender_key": key, "status": result["status"]})


def run_contender_process(payload: dict[str, Any], events: Any) -> None:
    """Обёртка для дочернего процесса: сделать работу и завершиться немедленно.

    Обычного возврата из функции недостаточно. `n_jobs`, взятый из бюджета ресурсов,
    заставляет sklearn поднять пул joblib, и после окончания работы процесс остаётся
    жить, пока пул не истечёт по своему таймауту — измерено: 2 секунды против больше
    тридцати на той же задаче. Планировщик при этом видит живой процесс и считает
    контендера незавершённым, хотя результат уже лежит на диске.

    Результат записан и события отправлены, поэтому штатное сворачивание интерпретатора
    ничего полезного больше не делает — и мы его не ждём.
    """
    run_contender(payload, events)
    _flush_events(events)
    os._exit(0)


def _flush_events(events: Any) -> None:
    if events is None:
        return

    try:
        events.close()
    except (OSError, ValueError):
        return

    #join_thread() не имеет предела ожидания, а ждать его бесконечно — то же зависание,
    #от которого мы уходим. Ограничиваем ожидание отдельным потоком
    joiner = threading.Thread(target=events.join_thread, daemon=True)
    joiner.start()
    joiner.join(EVENT_FLUSH_SECONDS)


def _failure_result(key: str, code: str, message: str, started: float) -> dict[str, Any]:
    return {
        "contender_key": key,
        "status": "FAILED",
        "error_code": code,
        "error_message": message,
        "folds": [],
        "warnings": [],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _emit(events: Any, message: dict[str, Any]) -> None:
    if events is None:
        return

    try:
        events.put(message)
    except (OSError, ValueError):
        #очередь закрыта, потому что родитель уже не слушает: сообщать некому,
        #но и падать из-за этого контендеру незачем
        return


def _write_traceback(staging: Path, detail: str) -> None:
    try:
        (staging / TRACEBACK_FILE).write_text(detail, encoding="utf-8")
    except OSError:
        return


def _train(payload: dict[str, Any], events: Any, staging: Path) -> dict[str, Any]:
    import numpy as np

    from backend.arena.predictions import (
        HOLDOUT_FOLD,
        SPLIT_CV,
        SPLIT_HOLDOUT,
        PredictionError,
        to_frame,
        validate_fold_predictions,
        verify_oof_coverage,
    )
    from backend.models.context import ResourceBudget

    key = payload["contender_key"]
    task_type = payload["task_type"]
    class_labels: list[str] = payload["class_labels"]
    budget = ResourceBudget(**payload["budget"])
    seed = int(payload["seed"])
    started = time.monotonic()

    splits = np.load(payload["folds_path"])
    row_ids = splits["row_ids"]
    features, target = _load_matrix(payload)

    if len(row_ids) != len(features):
        #индексы фолдов — это позиции в таблице строк с известной целью. Если снимок
        #дал другое число строк, все индексы указывают не туда, и метрика получится
        #правдоподобной по величине, но посчитанной по чужим строкам
        raise ContenderError(
            "incompatible_task",
            f"Снимок датасета дал {len(features)} строк, а разбиение построено на "
            f"{len(row_ids)}: фолды относятся к другой таблице.",
            detail="",
        )

    parts = []
    fold_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    n_folds = int(splits["n_folds"])

    for fold in range(n_folds):
        train_index = splits[f"train_{fold}"]
        validation_index = splits[f"val_{fold}"]
        #проверка границы, а не доверие вызывающему: пересечение обучения и проверки —
        #это метрика по строкам, которые модель уже видела
        overlap = np.intersect1d(train_index, validation_index)

        if overlap.size:
            raise ContenderError(
                "incompatible_task",
                f"Фолд {fold}: {overlap.size} строк одновременно в обучении и в проверке.",
                detail="",
            )

        fold_started = time.monotonic()
        prediction, missing = _fit_and_predict(
            payload=payload,
            budget=budget,
            seed=seed,
            features=features,
            target=target,
            row_ids=row_ids,
            train_index=train_index,
            predict_index=validation_index,
            fold=fold,
            split=SPLIT_CV,
        )
        validate_fold_predictions(
            prediction,
            expected_rows=len(validation_index),
            task_type=task_type,
            class_labels=class_labels,
        )
        parts.append(prediction)

        if missing:
            warnings.append(
                f"Фолд {fold}: классы {missing} отсутствовали в обучающей части, "
                "и вероятность по ним равна нулю — модель их не видела."
            )

        fold_reports.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "seconds": round(time.monotonic() - fold_started, 3),
            }
        )
        _emit(
            events,
            {
                "kind": "fold_completed",
                "contender_key": key,
                "fold": fold,
                "n_folds": n_folds,
                "seconds": fold_reports[-1]["seconds"],
            },
        )

    holdout_index = splits["holdout"]
    train_pool = splits["train_pool"]

    if holdout_index.size:
        #holdout считается один раз в самом конце и только как подтверждение: ни отбор
        #участников, ни подбор порога на него не опираются
        prediction, _ = _fit_and_predict(
            payload=payload,
            budget=budget,
            seed=seed,
            features=features,
            target=target,
            row_ids=row_ids,
            train_index=train_pool,
            predict_index=holdout_index,
            fold=HOLDOUT_FOLD,
            split=SPLIT_HOLDOUT,
        )
        validate_fold_predictions(
            prediction,
            expected_rows=len(holdout_index),
            task_type=task_type,
            class_labels=class_labels,
        )
        parts.append(prediction)

    frame = to_frame(parts, task_type=task_type, class_labels=class_labels)

    try:
        coverage = verify_oof_coverage(
            frame, expected_row_ids=row_ids[train_pool], strategy=payload["strategy"]
        )
    except PredictionError as error:
        raise ContenderError(error.code, error.message, detail="") from error

    try:
        frame.write_parquet(staging / PREDICTIONS_FILE, compression="zstd")
    except OSError as error:
        raise ContenderError(
            "artifact_write_failed",
            f"Предсказания не записались на диск: {type(error).__name__}.",
        ) from error

    return {
        "contender_key": key,
        "status": "SUCCEEDED",
        "error_code": None,
        "error_message": "",
        "folds": fold_reports,
        "coverage": coverage.to_dict(),
        "warnings": warnings,
        "holdout_rows": int(holdout_index.size),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _load_matrix(payload: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np
    import polars as pl

    from backend.preprocessing.profiles import normalize_categoricals, split_feature_types
    from backend.tasks.spec import label_strings

    target_column = payload["target_column"]
    frame = pl.read_parquet(payload["dataset_path"])
    #та же фильтрация, что при построении фолдов: индексы относятся именно к строкам
    #с известной целью, и любое другое подмножество сдвинуло бы их все
    usable = frame.filter(pl.col(target_column).is_not_null())
    features = usable.select(payload["feature_columns"]).to_pandas()
    types = split_feature_types(features)
    features = normalize_categoricals(features, types.categorical)

    if payload["task_type"] == "regression":
        return features, usable[target_column].cast(pl.Float64).to_numpy()

    #метки приводятся к строкам ровно тем же способом, что и в постановке задачи:
    #polars даёт «true», а str() — «True», и расхождение оставило бы положительный класс
    #ненайденным, ничего при этом не сломав явно
    return features, np.asarray(label_strings(usable[target_column]).to_list(), dtype=object)


def _fit_and_predict(
    *,
    payload: dict[str, Any],
    budget: Any,
    seed: int,
    features: Any,
    target: Any,
    row_ids: Any,
    train_index: Any,
    predict_index: Any,
    fold: int,
    split: str,
) -> tuple[Any, list[str]]:
    """Обучить на переданной части и предсказать на проверочной.

    Препроцессор и модель собираются **заново на каждый фолд**. Переиспользованный
    трансформер принёс бы в следующий фолд статистики предыдущего — ровно та утечка,
    против которой строился весь протокол. Всё обучение идёт внутри `Pipeline`, поэтому
    посчитать статистики до разбиения здесь физически негде: `fit` получает только
    строки обучающей части.
    """
    import numpy as np
    from joblib import parallel_config
    from sklearn.pipeline import Pipeline

    from backend.arena.predictions import FoldPredictions, align_probabilities
    from backend.models.registry import adapter_by_key
    from backend.preprocessing.profiles import build_preprocessor, split_feature_types

    task_type = payload["task_type"]
    train_features = features.iloc[train_index]
    predict_features = features.iloc[predict_index]

    try:
        #типы определяются по dtype колонки, а не по значениям: это не статистика,
        #и считать её по всей таблице безопасно
        types = split_feature_types(features)
        adapter = adapter_by_key(task_type, payload["adapter_key"])
        pipeline = Pipeline(
            [
                ("preprocessing", build_preprocessor(payload["preprocessing_profile"], types)),
                ("model", adapter.build(dict(payload["params"]), budget, seed)),
            ]
        )
    except (TypeError, ValueError) as error:
        raise ContenderError(
            "preprocessing_failed", f"Не удалось собрать конвейер: {type(error).__name__}."
        ) from error

    try:
        #`n_jobs` по умолчанию поднимает в sklearn пул ПРОЦЕССОВ. Внутри контендера,
        #который сам уже отдельный процесс, это даёт вложенный параллелизм: память
        #умножается на число заданий, а после работы пул переживает своего создателя
        #и остаётся висеть в системе. Бюджет обещает потоки — потоки и выдаём
        with parallel_config(backend="threading", n_jobs=max(1, budget.threads_per_fit)):
            pipeline.fit(train_features, target[train_index])
    except MemoryError as error:
        raise ContenderError(
            "resource_limit", "Не хватило памяти на обучение этого контендера."
        ) from error
    except Exception as error:
        #широкий перехват здесь осознан: внутри стороннего кода бывает что угодно,
        #а перечислить исключения xgboost, lightgbm и catboost заранее нельзя. Ошибка
        #не проглатывается — она превращается в типизированный отказ одного контендера
        #с полным стеком в файле прогона
        raise ContenderError(
            "training_failed", f"Обучение на фолде {fold} прервалось: {type(error).__name__}."
        ) from error

    try:
        predicted = np.asarray(pipeline.predict(predict_features))
    except Exception as error:
        raise ContenderError(
            "prediction_failed",
            f"Модель обучилась, но не смогла предсказать: {type(error).__name__}.",
        ) from error

    proba: Any = None
    missing: list[str] = []

    if task_type != "regression" and payload["supports_proba"]:
        try:
            raw = np.asarray(pipeline.predict_proba(predict_features))
        except (AttributeError, NotImplementedError):
            #модель объявила поддержку вероятностей, но не даёт их: метрики, которым
            #они нужны, окажутся «не посчитаны» с причиной, а не нулём
            raw = None

        if raw is not None:
            model_classes = [str(value) for value in pipeline.named_steps["model"].classes_]
            proba, missing = align_probabilities(raw, model_classes, payload["class_labels"])

    if task_type == "regression":
        prepared = np.asarray(predicted, dtype=np.float64)
    else:
        prepared = np.asarray([str(value) for value in predicted], dtype=object)

    return (
        FoldPredictions(
            row_ids=row_ids[predict_index],
            fold=fold,
            split=split,
            y_true=target[predict_index],
            y_pred=prepared,
            proba=proba,
        ),
        missing,
    )
