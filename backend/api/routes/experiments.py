"""Эксперименты: история прогонов и их сравнение.

Сравнение начинается не с метрик, а с перечня того, чем условия отличаются. Две метрики
рядом без упоминания разного разбиения — самый удобный способ обмануть себя, и слой
устроен так, чтобы этого не позволить.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.arena.runner import get_runner
from backend.experiments.history import compare, describe_run, list_experiments

router = APIRouter(prefix="/experiments", tags=["experiments"])


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2)


@router.get(
    "",
    summary="История экспериментов",
    description=(
        "Все прогоны с условиями, при которых они выполнялись: снимок данных, задача, "
        "протокол, состав участников, вердикт о чемпионе и сохранённые модели."
    ),
)
def get_experiments() -> dict[str, Any]:
    experiments = list_experiments(get_runner().store)
    return {"experiments": experiments, "total": len(experiments)}


@router.get("/{run_id}", summary="Карточка эксперимента")
def get_experiment(run_id: str) -> dict[str, Any]:
    return {"experiment": describe_run(get_runner().get(run_id))}


@router.post(
    "/compare",
    summary="Сравнить эксперименты",
    description=(
        "Показывает, чем отличаются условия прогонов, и только потом их результаты. "
        "Если отличается снимок данных, набор признаков или фактическое разбиение, "
        "прямое сравнение метрик названо некорректным."
    ),
)
def compare_experiments(request: CompareRequest) -> dict[str, Any]:
    return compare(get_runner().store, request.run_ids)
