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
from backend.datasets.registry import DatasetRegistry
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
