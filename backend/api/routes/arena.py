"""Прогоны: запуск, наблюдение, остановка.

Прогресс отдаётся опросом, а не потоком событий. Для честной картины этого достаточно:
состояние на сервере меняется по фактам — «фолд посчитан», «контендер завершён», — и опрос
возвращает ровно то, что уже случилось. Поток событий добавил бы задержку в доли секунды
и переподключения, не добавив достоверности.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from backend.arena.runner import get_runner
from backend.arena.store import RunRecord
from backend.core.errors import NotFoundError, ValidationError
from backend.datasets.registry import DatasetRegistry
from backend.diagnostics.errors import select_error_rows
from backend.diagnostics.report import build_diagnostics
from backend.diagnostics.thresholds import ERROR_ROWS_DEFAULT
from backend.leakage.report import inspect_leakage
from backend.models.context import ResourceBudget, build_context
from backend.models.registry import build_contenders
from backend.protocol.folds import build_folds
from backend.protocol.recommend import (
    DEFAULT_HOLDOUT_SIZE,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    recommend_protocol,
)
from backend.readiness.report import assess_readiness
from backend.scoring.leaderboard import build_leaderboard
from backend.scoring.objective import build_objective
from backend.tasks.spec import build_task_spec, propose_task_setup

router = APIRouter(prefix="/arena", tags=["arena"])


class StartRunRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    group_column: str | None = None
    time_column: str | None = None
    n_splits: int = DEFAULT_N_SPLITS
    holdout_size: float = DEFAULT_HOLDOUT_SIZE
    seed: int = DEFAULT_SEED
    #None означает «весь подходящий состав». Baseline остаётся в наборе при любом выборе
    selected_contenders: list[str] | None = None


@router.post(
    "/runs",
    summary="Запустить прогон",
    description=(
        "Обучает выбранных участников по одному утверждённому протоколу. Возвращается сразу: "
        "состояние читается опросом карточки. Повторная отправка того же эксперимента, пока он "
        "идёт, возвращает уже запущенный прогон, а не создаёт второй."
    ),
)
def start_run(request: StartRunRequest, response: Response) -> dict[str, Any]:
    registry = DatasetRegistry()
    snapshot = registry.get(request.dataset_id)
    frame = registry.load_frame(request.dataset_id)

    proposal = propose_task_setup(snapshot, frame, request.target_column)
    task = build_task_spec(
        frame,
        target_column=proposal.target_column,
        task_type=proposal.task_type,
        feature_columns=proposal.feature_columns,
        positive_label=proposal.positive_label,
        group_column=request.group_column,
        time_column=request.time_column,
    )
    protocol = recommend_protocol(
        frame,
        task,
        n_splits=request.n_splits,
        holdout_size=request.holdout_size,
        seed=request.seed,
    )
    #разбиение строится один раз и становится законом для всех участников: одинаковые
    #фолды — это и есть техническая гарантия честности сравнения (D-7)
    folds = build_folds(frame, task, protocol)
    leakage = inspect_leakage(frame, task, protocol, folds)
    readiness = assess_readiness(snapshot, frame, task, protocol, leakage)
    budget = ResourceBudget.detect()
    context = build_context(frame, task, n_train_rows=len(folds.train_pool))
    contenders = build_contenders(context, selected_keys=request.selected_contenders)

    record, duplicate = get_runner().start(
        dataset_id=request.dataset_id,
        dataset_fingerprint=snapshot.fingerprint,
        dataset_path=registry.data_path(request.dataset_id),
        frame=frame,
        task=task,
        protocol=protocol,
        folds=folds,
        contenders=contenders,
        budget=budget,
        analysis={"readiness": readiness.to_dict(), "leakage": leakage.to_dict()},
    )
    #200 на повторную отправку и 202 на новый прогон: клиент по коду отличает
    #«приняли к работе» от «это тот же самый, что уже идёт»
    response.status_code = 200 if duplicate else 202
    return {"run": _view(record), "duplicate_of_active_run": duplicate}


@router.get("/runs", summary="История прогонов")
def list_runs() -> dict[str, Any]:
    return {"runs": [_summary(record) for record in get_runner().list_runs()]}


@router.get("/runs/{run_id}", summary="Карточка прогона")
def get_run(run_id: str) -> dict[str, Any]:
    runner = get_runner()
    record = runner.get(run_id)
    return {
        "run": _view(record),
        "progress": runner.progress(run_id).to_dict(),
        "active": runner.is_active(run_id),
    }


@router.get("/runs/{run_id}/contenders", summary="Участники прогона")
def get_contenders(run_id: str) -> dict[str, Any]:
    record = get_runner().get(run_id)
    #упавшие и пропущенные остаются в списке: исчезнув, они выглядели бы проигравшими
    return {"contenders": [item.to_dict() for item in record.contenders]}


class ConstraintRequest(BaseModel):
    metric: str = Field(min_length=1)
    operator: str = "gte"
    value: float
    #зачем это ограничение: без объяснения отсев участника выглядит произволом
    reason: str = ""


class ObjectiveRequest(BaseModel):
    #None означает «цель по умолчанию для этого типа задачи»
    metric: str | None = None
    constraints: list[ConstraintRequest] = Field(default_factory=list)
    min_gain_over_baseline: float = 0.0


@router.get(
    "/runs/{run_id}/leaderboard",
    summary="Таблица результатов по цели по умолчанию",
    description=(
        "Пересчитывается из сохранённых предсказаний, а не читается из карточки прогона: "
        "смена цели сравнения не требует переобучения. Ранжирование идёт по кросс-валидации, "
        "holdout показан рядом и в выборе не участвует."
    ),
)
def get_leaderboard(run_id: str, metric: str | None = None) -> dict[str, Any]:
    return _leaderboard(run_id, ObjectiveRequest(metric=metric))


@router.post(
    "/runs/{run_id}/leaderboard",
    summary="Таблица результатов по заданной цели",
    description=(
        "Та же таблица под другой целью: другая метрика, ограничения пригодности, "
        "минимальный выигрыш над точкой отсчёта. Обучение не повторяется."
    ),
)
def post_leaderboard(run_id: str, request: ObjectiveRequest) -> dict[str, Any]:
    return _leaderboard(run_id, request)


def _leaderboard(run_id: str, request: ObjectiveRequest) -> dict[str, Any]:
    runner = get_runner()
    record = runner.get(run_id)
    task_type = str((record.spec.get("task") or {}).get("task_type", ""))
    objective = build_objective(
        task_type,
        metric=request.metric,
        constraints=[item.model_dump() for item in request.constraints],
        min_gain_over_baseline=request.min_gain_over_baseline,
    )
    leaderboard = build_leaderboard(
        record=record, store=runner.store, objective=objective
    )
    return {"run_id": run_id, "run_state": record.state, "leaderboard": leaderboard.to_dict()}


@router.get(
    "/runs/{run_id}/contenders/{contender_key}/diagnostics",
    summary="Диагностика участника",
    description=(
        "Что происходит с моделью: разрыв между обучающей и проверочной частью, "
        "устойчивость по фолдам, разбор по классам либо остатки для регрессии. "
        "Отвечает на другой вопрос, чем таблица результатов, и пороги отдаёт наружу."
    ),
)
def get_diagnostics(run_id: str, contender_key: str, metric: str | None = None) -> dict[str, Any]:
    runner = get_runner()
    record = runner.get(run_id)
    task = _task_of(record)
    frame = _predictions_of(runner, run_id, contender_key)

    diagnostics = build_diagnostics(
        frame,
        contender_key=contender_key,
        task_type=str(task.get("task_type", "")),
        class_labels=list(task.get("class_labels") or []),
        positive_label=task.get("positive_label"),
        metric=metric,
    )
    return {"run_id": run_id, "diagnostics": diagnostics.to_dict()}


@router.get(
    "/runs/{run_id}/contenders/{contender_key}/errors",
    summary="Строки, на которых модель ошиблась",
    description=(
        "Переход от «модель ошибается» к конкретным наблюдениям: идентификатор строки "
        "плюс исходные значения признаков из того же неизменяемого снимка."
    ),
)
def get_error_rows(  # noqa: PLR0917
    run_id: str,
    contender_key: str,
    kind: str = "all_mistakes",
    actual: str | None = None,
    predicted: str | None = None,
    limit: int = ERROR_ROWS_DEFAULT,
) -> dict[str, Any]:
    runner = get_runner()
    record = runner.get(run_id)
    task = _task_of(record)
    frame = _predictions_of(runner, run_id, contender_key)

    return {
        "run_id": run_id,
        "contender_key": contender_key,
        "errors": select_error_rows(
            frame,
            dataset_id=str(record.spec.get("dataset_id", "")),
            task_type=str(task.get("task_type", "")),
            class_labels=list(task.get("class_labels") or []),
            positive_label=task.get("positive_label"),
            feature_columns=list(task.get("feature_columns") or []),
            kind=kind,  # type: ignore[arg-type]
            actual=actual,
            predicted=predicted,
            limit=limit,
        ),
    }


def _task_of(record: RunRecord) -> dict[str, Any]:
    return dict(record.spec.get("task") or {})


def _predictions_of(runner: Any, run_id: str, contender_key: str) -> Any:
    from backend.arena.metrics import read_predictions

    known = {item.contender_key for item in runner.get(run_id).contenders}

    if contender_key not in known:
        raise NotFoundError(f"В прогоне {run_id} нет участника «{contender_key}».")

    path = runner.store.predictions_path(run_id, contender_key)

    if not path.exists():
        raise NotFoundError(
            f"У участника «{contender_key}» нет сохранённых предсказаний: "
            "он не завершился успешно."
        )

    try:
        return read_predictions(path)
    except (OSError, ValueError) as error:
        raise ValidationError(f"Предсказания не читаются: {type(error).__name__}.") from error


@router.get(
    "/runs/{run_id}/contenders/{contender_key}/learning-curve",
    summary="Кривая обучения участника",
    description=(
        "Обучает участника на растущих долях той же обучающей части и проверяет на той же "
        "проверочной. Отвечает измерением на вопрос «хватает ли данных». Вычисление "
        "синхронное и ограниченное: стоимость в числе обучений возвращается вместе "
        "с результатом."
    ),
)
def get_learning_curve(
    run_id: str, contender_key: str, metric: str | None = None
) -> dict[str, Any]:
    from backend.datasets.registry import DatasetRegistry
    from backend.diagnostics.learning_curve import build_learning_curve

    runner = get_runner()
    record = runner.get(run_id)
    contender = next(
        (
            item
            for item in record.spec.get("contenders", [])
            if item.get("contender_key") == contender_key
        ),
        None,
    )

    if contender is None:
        raise NotFoundError(f"В прогоне {run_id} нет участника «{contender_key}».")

    folds_path = runner.store.folds_path(run_id)

    if not folds_path.exists():
        raise NotFoundError(
            "Разбиение прогона не сохранено: кривую обучения построить не на чем."
        )

    registry = DatasetRegistry()
    dataset_id = str(record.spec.get("dataset_id", ""))

    return {
        "run_id": run_id,
        "contender_key": contender_key,
        "learning_curve": build_learning_curve(
            folds_path=folds_path,
            dataset_path=registry.data_path(dataset_id),
            task=dict(record.spec.get("task") or {}),
            contender=contender,
            seed=int(record.spec.get("seed", 0)),
            threads=int((record.spec.get("budget") or {}).get("threads_per_fit", 1)),
            metric=metric,
        ),
    }


@router.get("/runs/{run_id}/events", summary="Журнал событий прогона")
def get_events(run_id: str, since: int = 0) -> dict[str, Any]:
    events = get_runner().events(run_id, since=max(0, since))
    return {"since": since, "events": [event.to_dict() for event in events]}


@router.post("/runs/{run_id}/cancel", summary="Остановить прогон")
def cancel_run(run_id: str) -> dict[str, Any]:
    record = get_runner().cancel(run_id)
    return {"run": _view(record)}


def _view(record: RunRecord) -> dict[str, Any]:
    payload = record.to_dict()
    #метка процесса-владельца — внутреннее дело backend: наружу она ничего не объясняет
    payload.pop("owner_token", None)
    return payload


def _summary(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "state": record.state,
        "error_code": record.error_code,
        "dataset_id": record.spec.get("dataset_id"),
        "task_type": (record.spec.get("task") or {}).get("task_type"),
        "target_column": (record.spec.get("task") or {}).get("target_column"),
        "experiment_fingerprint": record.experiment_fingerprint,
        "contenders": {
            "planned": len(record.contenders),
            "succeeded": sum(1 for item in record.contenders if item.state == "SUCCEEDED"),
            "failed": sum(1 for item in record.contenders if item.state == "FAILED"),
            "skipped": sum(1 for item in record.contenders if item.state == "SKIPPED"),
        },
    }
