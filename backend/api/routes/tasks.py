"""Постановка задачи: разбор до обучения."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

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

router = APIRouter(prefix="/datasets", tags=["tasks"])


class AnalyzeRequest(BaseModel):
    target_column: str = Field(min_length=1)
    group_column: str | None = None
    time_column: str | None = None
    n_splits: int = DEFAULT_N_SPLITS
    holdout_size: float = DEFAULT_HOLDOUT_SIZE
    seed: int = DEFAULT_SEED


@router.post(
    "/{dataset_id}/task/analyze",
    summary="Разобрать постановку задачи без обучения",
    description=(
        "Определяет тип задачи, предлагает признаки и исключения с причинами, "
        "рекомендует протокол оценки и оценивает готовность тройки данные+задача+протокол. "
        "Объясняет каждое решение числами и ничего не обучает."
    ),
)
def analyze_task(dataset_id: str, request: AnalyzeRequest) -> dict[str, Any]:
    registry = DatasetRegistry()
    snapshot = registry.get(dataset_id)
    frame = registry.load_frame(dataset_id)

    proposal = propose_task_setup(snapshot, frame, request.target_column)
    #протокол считается на предложенной постановке: пользователь увидит рекомендацию
    #до того, как потратит время на обучение
    spec = build_task_spec(
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
        spec,
        n_splits=request.n_splits,
        holdout_size=request.holdout_size,
        seed=request.seed,
    )

    #проверки утечки идут по ФАКТИЧЕСКИМ границам фолдов, поэтому разбиение строится здесь
    folds = build_folds(frame, spec, protocol)
    leakage = inspect_leakage(frame, spec, protocol, folds)
    #готовность оценивается по фактической тройке и включает находки утечки (D-22, D-23)
    readiness = assess_readiness(snapshot, frame, spec, protocol, leakage)

    #состав участников виден ДО запуска: недоступный адаптер показывается со статусом
    #и текстом, что поставить, а не исчезает из списка (D-2)
    budget = ResourceBudget.detect()
    context = build_context(frame, spec, n_train_rows=len(folds.train_pool))
    contenders = build_contenders(context)

    return {
        "task": proposal.to_dict(),
        "protocol": protocol.to_dict(),
        "folds": folds.to_dict(),
        "leakage": leakage.to_dict(),
        "readiness": readiness.to_dict(),
        "contenders": [item.to_dict() for item in contenders],
        "resources": budget.to_dict(),
    }
