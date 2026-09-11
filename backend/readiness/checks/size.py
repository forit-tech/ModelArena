"""Хватает ли наблюдений на то число признаков, которое реально получит модель."""
from __future__ import annotations

import polars as pl

from backend.readiness.report import (
    COMFORTABLE_ROWS_PER_DIMENSION,
    COMFORTABLE_VALIDATION_ROWS,
    CRITICAL_ROWS_PER_DIMENSION,
    CRITICAL_VALIDATION_ROWS,
    Finding,
    ReadinessContext,
)
from backend.tasks.spec import TaskSpec


def check(frame: pl.DataFrame, spec: TaskSpec, context: ReadinessContext) -> list[Finding]:  # noqa: ARG001
    findings: list[Finding] = []
    ratio = context.rows_per_dimension

    #900 строк на 4 признака и 900 строк на 500 признаков — принципиально разные ситуации,
    #поэтому порог задаётся отношением, а не абсолютным числом строк
    if ratio < CRITICAL_ROWS_PER_DIMENSION:
        findings.append(
            Finding(
                code="sample_size_vs_dimension",
                severity="high_risk",
                scope="dataset",
                observed_value=ratio,
                threshold=(
                    f"{CRITICAL_ROWS_PER_DIMENSION} наблюдений на признак — нижняя граница, "
                    f"{COMFORTABLE_ROWS_PER_DIMENSION} — комфортная"
                ),
                explanation=(
                    f"На {context.usable_rows} строк приходится {context.effective_dimension} "
                    f"признаков после кодирования: {ratio} наблюдения на признак."
                ),
                consequence=(
                    "Гибкие модели запомнят обучающую выборку целиком, разброс между фолдами "
                    "перекроет разницу между моделями, и leaderboard будет ранжировать шум."
                ),
                suggested_action=(
                    "Сократите набор признаков, объедините редкие категории "
                    "или соберите больше наблюдений."
                ),
            )
        )
    elif ratio < COMFORTABLE_ROWS_PER_DIMENSION:
        findings.append(
            Finding(
                code="sample_size_vs_dimension",
                severity="caution",
                scope="dataset",
                observed_value=ratio,
                threshold=f"{COMFORTABLE_ROWS_PER_DIMENSION} наблюдений на признак",
                explanation=(
                    f"{ratio} наблюдений на признак после кодирования "
                    f"({context.effective_dimension} измерений на {context.usable_rows} строк)."
                ),
                consequence="Разница между близкими моделями может оказаться в пределах шума.",
                suggested_action=(
                    "Сравнивайте модели по CV со стандартным отклонением, а не по одному числу."
                ),
            )
        )

    if context.validation_rows_per_fold < CRITICAL_VALIDATION_ROWS:
        findings.append(
            Finding(
                code="validation_fold_too_small",
                severity="high_risk",
                scope="protocol",
                observed_value=context.validation_rows_per_fold,
                threshold=f"{COMFORTABLE_VALIDATION_ROWS} строк на проверочную часть фолда",
                explanation=(
                    f"На проверочную часть фолда приходится около "
                    f"{context.validation_rows_per_fold} строк."
                ),
                consequence=(
                    "Одна-две ошибки меняют метрику на десятки процентов: сравнение моделей "
                    "перестаёт что-либо измерять."
                ),
                suggested_action="Уменьшите число фолдов или соберите больше данных.",
            )
        )
    elif context.validation_rows_per_fold < COMFORTABLE_VALIDATION_ROWS:
        findings.append(
            Finding(
                code="validation_fold_small",
                severity="caution",
                scope="protocol",
                observed_value=context.validation_rows_per_fold,
                threshold=f"{COMFORTABLE_VALIDATION_ROWS} строк на проверочную часть фолда",
                explanation=f"В проверочной части фолда около {context.validation_rows_per_fold} строк.",
                consequence="Метрика по фолдам будет заметно колебаться.",
                suggested_action="Ориентируйтесь на среднее по фолдам вместе с разбросом.",
            )
        )

    return findings
