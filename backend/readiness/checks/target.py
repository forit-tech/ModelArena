"""Пригодна ли цель к тому, чтобы по ней вообще что-то сравнивать."""
from __future__ import annotations

import polars as pl

from backend.protocol.recommend import ProtocolProposal
from backend.readiness.report import (
    COMFORTABLE_MINORITY_PER_FOLD,
    IMBALANCE_CAUTION,
    IMBALANCE_HIGH_RISK,
    RISKY_MINORITY_PER_FOLD,
    TARGET_MISSING_CAUTION,
    TARGET_MISSING_HIGH_RISK,
    Finding,
    ReadinessContext,
)
from backend.tasks.spec import TaskSpec


def check(
    frame: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
    context: ReadinessContext,
) -> list[Finding]:
    findings = list(_missing_target(frame, spec, context))

    if spec.task_type == "regression":
        findings.extend(_regression_target(frame, spec))
    else:
        findings.extend(_class_balance(frame, spec, protocol, context))

    return findings


def _missing_target(frame: pl.DataFrame, spec: TaskSpec, context: ReadinessContext) -> list[Finding]:  # noqa: ARG001
    missing = context.total_rows - context.usable_rows

    if missing == 0:
        return []

    ratio = missing / context.total_rows
    #пропуск в цели — не «плохое качество данных», а прямая потеря наблюдений:
    #такие строки не участвуют ни в обучении, ни в оценке
    severity = (
        "high_risk"
        if ratio >= TARGET_MISSING_HIGH_RISK
        else "caution"
        if ratio >= TARGET_MISSING_CAUTION
        else "info"
    )

    return [
        Finding(
            code="target_missing_values",
            severity=severity,
            scope="target",
            observed_value=round(ratio, 4),
            threshold=f"{TARGET_MISSING_CAUTION:.0%} — внимание, {TARGET_MISSING_HIGH_RISK:.0%} — риск",
            explanation=(
                f"У {missing} из {context.total_rows} строк цель «{spec.target_column}» пуста "
                f"({ratio:.0%})."
            ),
            consequence=(
                "Эти строки выпадают из обучения и из оценки. Если пропуск в цели связан "
                "с самим исходом, оставшаяся выборка смещена, и сравнение проводится "
                "не на той популяции, для которой модель предназначена."
            ),
            suggested_action=(
                "Проверьте в DataArena, почему цель пуста именно у этих строк. "
                "Случайный пропуск безопасен, систематический — нет."
            ),
        )
    ]


def _class_balance(
    frame: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
    context: ReadinessContext,
) -> list[Finding]:
    counts = frame[spec.target_column].drop_nulls().value_counts(sort=True)
    frequencies = counts["count"].to_list()
    labels = [str(value) for value in counts[spec.target_column].to_list()]

    if not frequencies:
        return []

    findings: list[Finding] = []
    dominant_share = frequencies[0] / context.usable_rows
    minority = int(min(frequencies))
    minority_label = labels[frequencies.index(minority)]
    per_fold = minority / protocol.n_splits if protocol.n_splits else 0

    if dominant_share >= IMBALANCE_HIGH_RISK:
        findings.append(
            Finding(
                code="target_almost_constant",
                severity="high_risk",
                scope="target",
                observed_value=round(dominant_share, 4),
                threshold=f"{IMBALANCE_HIGH_RISK:.0%} доли одного класса",
                explanation=f"Класс «{labels[0]}» занимает {dominant_share:.1%} строк — цель почти постоянна.",
                consequence=(
                    "Модель, всегда отвечающая большинством, получит почти идеальную accuracy, "
                    "и превзойти её осмысленно будет невозможно: сравнивать окажется нечего."
                ),
                suggested_action=(
                    "Выберите другую цель или другой горизонт события, где редкий класс "
                    "встречается чаще."
                ),
                blocking=True,
            )
        )
    elif dominant_share >= IMBALANCE_CAUTION:
        findings.append(
            Finding(
                code="target_imbalanced",
                severity="caution",
                scope="target",
                observed_value=round(dominant_share, 4),
                threshold=f"{IMBALANCE_CAUTION:.0%} доли одного класса",
                explanation=f"Класс «{labels[0]}» занимает {dominant_share:.0%} строк.",
                consequence=(
                    f"Accuracy здесь вводит в заблуждение: постоянный ответ «{labels[0]}» "
                    f"уже даёт {dominant_share:.0%}. Ранжирование моделей по ней будет ложным."
                ),
                suggested_action=(
                    "Читайте leaderboard по PR-AUC, balanced accuracy и recall редкого класса, "
                    "а не по accuracy."
                ),
            )
        )

    if per_fold < 1:
        findings.append(
            Finding(
                code="minority_class_absent_in_folds",
                severity="high_risk",
                scope="target",
                observed_value=minority,
                threshold=f"минимум {protocol.n_splits} объектов на {protocol.n_splits} фолдов",
                explanation=(
                    f"Класс «{minority_label}» встречается {minority} раз при "
                    f"{protocol.n_splits} фолдах."
                ),
                consequence=(
                    "В части фолдов этого класса не окажется вовсе: метрики по нему "
                    "будут неопределены, а среднее по фолдам — посчитано по разным задачам."
                ),
                suggested_action="Уменьшите число фолдов, объедините классы или соберите данные.",
                blocking=True,
            )
        )
    elif per_fold < RISKY_MINORITY_PER_FOLD:
        findings.append(
            Finding(
                code="minority_class_thin",
                severity="high_risk",
                scope="target",
                observed_value=round(per_fold, 2),
                threshold=f"{COMFORTABLE_MINORITY_PER_FOLD} объектов редкого класса на фолд",
                explanation=(
                    f"На фолд приходится около {per_fold:.1f} объектов класса «{minority_label}» "
                    f"({minority} на {protocol.n_splits} фолдов)."
                ),
                consequence=(
                    "Recall по этому классу будет прыгать на десятки процентов между фолдами, "
                    "и разницу между моделями по нему сравнивать нельзя."
                ),
                suggested_action=(
                    "Уменьшите число фолдов либо признайте, что по этому классу сравнение "
                    "недостоверно, и опирайтесь на метрики по всем классам."
                ),
            )
        )
    elif per_fold < COMFORTABLE_MINORITY_PER_FOLD:
        findings.append(
            Finding(
                code="minority_class_small",
                severity="caution",
                scope="target",
                observed_value=round(per_fold, 2),
                threshold=f"{COMFORTABLE_MINORITY_PER_FOLD} объектов редкого класса на фолд",
                explanation=f"На фолд приходится около {per_fold:.1f} объектов «{minority_label}».",
                consequence="Метрики по редкому классу будут заметно колебаться между фолдами.",
                suggested_action="Смотрите разброс по фолдам, а не только среднее.",
            )
        )

    return findings


def _regression_target(frame: pl.DataFrame, spec: TaskSpec) -> list[Finding]:
    values = frame[spec.target_column].drop_nulls().cast(pl.Float64)

    if values.len() == 0:
        return []

    std = values.std() or 0.0
    mean = values.mean() or 0.0

    #цель без разброса не даёт чему учиться: R² не определён, и все модели равны
    if std == 0:
        return [
            Finding(
                code="regression_target_constant",
                severity="high_risk",
                scope="target",
                observed_value=0.0,
                threshold="стандартное отклонение больше нуля",
                explanation=f"Все значения «{spec.target_column}» одинаковы.",
                consequence="R² не определён, а любая модель даёт одинаковый результат.",
                suggested_action="Выберите другую цель.",
                blocking=True,
            )
        ]

    relative = abs(std / mean) if mean else None

    if relative is not None and relative < 0.01:
        return [
            Finding(
                code="regression_target_low_variance",
                severity="caution",
                scope="target",
                observed_value=round(relative, 5),
                threshold="разброс не меньше 1% от среднего",
                explanation=(
                    f"Разброс «{spec.target_column}» составляет {relative:.2%} от среднего "
                    f"(σ = {std:.4g})."
                ),
                consequence=(
                    "Почти вся дисперсия — шум измерения. R² будет близок к нулю у всех моделей, "
                    "и различить их окажется нечем."
                ),
                suggested_action=(
                    "Проверьте, та ли это величина, и нет ли смысла предсказывать её изменение."
                ),
            )
        ]

    return []
