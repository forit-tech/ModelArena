"""История экспериментов: от разрозненных прогонов к сравнимым записям.

Один прогон с десятью участниками и десять прогонов с разными протоколами — разные вещи,
и путать их нельзя. Внутри прогона участники сравнимы по построению: одни фолды, один seed,
один набор признаков. **Между** прогонами это неверно: если сменился протокол, признаки
или сам снимок данных, числа сопоставимы лишь настолько, насколько совпадают условия.

Поэтому сравнение прогонов начинается не с метрик, а с перечня того, **чем условия
отличаются**. Показать две метрики рядом, умолчав о разном разбиении, — это и есть тот
способ обмануть себя, ради борьбы с которым весь остальной слой и написан.

Смена цели без переобучения новым экспериментом не считается. Обучение при этом
не происходило, и выдавать пересчёт таблицы за новый прогон значило бы потерять
происхождение решения.
"""
from __future__ import annotations

from typing import Any

from backend.arena.store import RunRecord, RunStore
from backend.core.errors import ValidationError

#сколько прогонов можно сравнивать за раз: больше на экран всё равно не помещается
MAX_COMPARED = 6


def describe_run(record: RunRecord) -> dict[str, Any]:
    """Карточка эксперимента для списка и сравнения."""
    spec = record.spec or {}
    task = dict(spec.get("task") or {})
    protocol = dict(spec.get("protocol") or {})
    contenders = record.contenders
    succeeded = [item for item in contenders if item.state == "SUCCEEDED"]

    return {
        "run_id": record.run_id,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "state": record.state,
        "error_code": record.error_code,
        "runtime_seconds": _runtime(record),
        "dataset": {
            "dataset_id": spec.get("dataset_id"),
            #отпечаток данных: два прогона на «том же» датасете сравнимы, только если он совпал
            "fingerprint": spec.get("dataset_fingerprint"),
        },
        "task": {
            "task_type": task.get("task_type"),
            "target_column": task.get("target_column"),
            "n_features": len(task.get("feature_columns") or []),
            "feature_columns": list(task.get("feature_columns") or []),
            "positive_label": task.get("positive_label"),
            "class_labels": list(task.get("class_labels") or []),
        },
        "protocol": {
            "cv_splitter": protocol.get("cv_splitter"),
            "n_splits": spec.get("n_splits"),
            "holdout_rows": spec.get("holdout_rows"),
            "train_pool_rows": spec.get("train_pool_rows"),
            "seed": spec.get("seed"),
            #отпечаток фактического разбиения: совпал — значит фолды буквально те же
            "fold_assignment_hash": spec.get("fold_assignment_hash"),
        },
        "experiment_fingerprint": spec.get("experiment_fingerprint"),
        "contenders": {
            "planned": len(contenders),
            "succeeded": len(succeeded),
            "failed": sum(1 for item in contenders if item.state == "FAILED"),
            "skipped": sum(1 for item in contenders if item.state == "SKIPPED"),
            "cancelled": sum(1 for item in contenders if item.state == "CANCELLED"),
            "keys": [item.contender_key for item in contenders],
        },
        #вердикт, посчитанный по цели по умолчанию в момент завершения. Пересчёт под
        #другую цель — это чтение тех же предсказаний, а не новый эксперимент
        "champion": record.summary or None,
        "artifacts": [
            {
                "contender_key": item.contender_key,
                "model_sha256": (item.artifact or {}).get("model_sha256"),
                "model_bytes": (item.artifact or {}).get("model_bytes"),
            }
            for item in succeeded
            if item.artifact
        ],
        "environment": spec.get("environment") or {},
    }


def list_experiments(store: RunStore) -> list[dict[str, Any]]:
    return [describe_run(record) for record in store.list_runs()]


def compare(store: RunStore, run_ids: list[str]) -> dict[str, Any]:
    """Сравнить эксперименты, начав с того, чем отличаются условия."""
    if len(run_ids) < 2:
        raise ValidationError("Для сравнения нужно минимум два прогона.")

    if len(run_ids) > MAX_COMPARED:
        raise ValidationError(f"За раз сравнивается не больше {MAX_COMPARED} прогонов.")

    records = [describe_run(store.get(run_id)) for run_id in run_ids]
    differences = _differences(records)

    return {
        "runs": records,
        "differences": differences,
        "comparable": not differences,
        "verdict": _verdict(differences),
    }


def _runtime(record: RunRecord) -> float | None:
    from datetime import datetime

    if not record.started_at or not record.finished_at:
        return None

    try:
        started = datetime.fromisoformat(record.started_at)
        finished = datetime.fromisoformat(record.finished_at)
    except ValueError:
        return None

    return round((finished - started).total_seconds(), 3)


def _differences(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Чем условия прогонов отличаются. Пустой список означает, что они совпадают."""
    checks = [
        ("dataset_fingerprint", "отпечаток датасета", lambda item: item["dataset"]["fingerprint"],
         "Данные разные: метрики относятся к разным наборам строк и напрямую несравнимы."),
        ("target_column", "целевая колонка", lambda item: item["task"]["target_column"],
         "Модели предсказывают разное — сравнивать их метрики бессмысленно."),
        ("task_type", "тип задачи", lambda item: item["task"]["task_type"],
         "Разные типы задач считаются разными метриками."),
        ("feature_columns", "набор признаков", lambda item: tuple(item["task"]["feature_columns"]),
         "Модели видели разный вход: разница в метрике может объясняться им, а не моделью."),
        ("fold_assignment_hash", "фактическое разбиение",
         lambda item: item["protocol"]["fold_assignment_hash"],
         "Фолды разные. Разница в метрике частично объясняется разбиением, "
         "и пофолдное сравнение между этими прогонами невозможно."),
        ("seed", "seed", lambda item: item["protocol"]["seed"],
         "Случайность задана по-разному: часть различия — это она."),
    ]
    differences: list[dict[str, Any]] = []

    for key, label, extract, consequence in checks:
        values = [extract(item) for item in records]

        if len({str(value) for value in values}) > 1:
            differences.append(
                {
                    "key": key,
                    "label": label,
                    "values": [
                        {"run_id": item["run_id"], "value": _plain(extract(item))}
                        for item in records
                    ],
                    "consequence": consequence,
                }
            )

    return differences


def _verdict(differences: list[dict[str, Any]]) -> str:
    if not differences:
        return (
            "Условия совпадают: те же данные, та же задача, тот же набор признаков "
            "и то же фактическое разбиение. Метрики этих прогонов сопоставимы напрямую."
        )

    return (
        "Условия отличаются по пунктам: "
        + ", ".join(item["label"] for item in differences)
        + ". Метрики можно смотреть рядом, но разница между ними объясняется не только "
        "моделями. Прямое сравнение чисел здесь было бы выводом, которого данные не дают."
    )


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)

    return value
