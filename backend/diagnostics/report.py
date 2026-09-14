"""Диагностика одной модели: что с ней происходит.

Отвечает на другой вопрос, чем таблица результатов. Таблица говорит, **какую модель
выбрать по объявленной цели**; диагностика говорит, **что с выбранной моделью не так** —
и это разные вопросы, которые нельзя смешивать в одном экране.

Ни одного вывода без чисел. Фраза «модель переобучена» без измерения — это ощущение,
выданное за результат; вместо неё здесь всегда стоит «на обучающей части F1 = 0.98,
на проверочной 0.81, разрыв 0.17, порог 0.15».

Разрыв между обучающей и проверочной частью **измеряется**, а не предполагается: worker
сохраняет предсказания на подвыборке обучающей части фолда тем же конвейером, которым
считал проверочную. Эти строки модель видела при обучении, поэтому результат по ним завышен
по построению — именно поэтому они и годятся как верхняя опора разрыва, и именно поэтому
в ранжировании не участвуют никогда.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from backend.arena.metrics import (
    MetricSet,
    direction_of,
    evaluate,
    label_of,
    metric_keys,
    primary_metric,
)
from backend.arena.predictions import SPLIT_CV, SPLIT_TRAIN_PROBE
from backend.diagnostics import thresholds

SEVERITY_ORDER = {"high": 0, "caution": 1, "info": 2}


@dataclass
class Finding:
    """Наблюдение с числами, которыми оно обосновано."""

    code: str
    severity: str
    title: str
    #измерения, на которых основан вывод: без них это мнение, а не находка
    evidence: dict[str, Any]
    explanation: str
    consequence: str
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "consequence": self.consequence,
            "suggestion": self.suggestion,
        }


@dataclass
class FoldRow:
    fold: int
    validation_rows: int
    validation_score: float | None
    probe_rows: int
    train_score: float | None
    gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "validation_rows": self.validation_rows,
            "validation_score": self.validation_score,
            "probe_rows": self.probe_rows,
            "train_score": self.train_score,
            "gap": self.gap,
        }


@dataclass
class Diagnostics:
    contender_key: str
    task_type: str
    metric: str
    metric_label: str
    higher_is_better: bool
    folds: list[FoldRow]
    mean_validation: float | None
    spread_validation: float | None
    mean_gap: float | None
    findings: list[Finding]
    thresholds: dict[str, Any]
    classification: dict[str, Any] | None = None
    regression: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contender_key": self.contender_key,
            "task_type": self.task_type,
            "metric": self.metric,
            "metric_label": self.metric_label,
            "higher_is_better": self.higher_is_better,
            "folds": [item.to_dict() for item in self.folds],
            "mean_validation": self.mean_validation,
            "spread_validation": self.spread_validation,
            "mean_gap": self.mean_gap,
            "findings": [item.to_dict() for item in self.findings],
            "thresholds": self.thresholds,
            "classification": self.classification,
            "regression": self.regression,
            "notes": self.notes,
        }


def build_diagnostics(
    frame: pl.DataFrame,
    *,
    contender_key: str,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
    metric: str | None = None,
) -> Diagnostics:
    """Собрать диагностику по сохранённым предсказаниям одного участника."""
    chosen = metric or primary_metric(task_type)

    if chosen not in metric_keys(task_type):
        chosen = primary_metric(task_type)

    higher = direction_of(chosen) == "higher"
    validation = frame.filter(pl.col("split") == SPLIT_CV)
    probe = frame.filter(pl.col("split") == SPLIT_TRAIN_PROBE)
    notes: list[str] = []

    folds = _fold_rows(
        validation=validation,
        probe=probe,
        task_type=task_type,
        class_labels=class_labels,
        positive_label=positive_label,
        metric=chosen,
        higher_is_better=higher,
    )

    if probe.height == 0:
        notes.append(
            "Замер на обучающей части отсутствует: разрыв train-vs-validation по этому "
            "прогону измерить нечем. Предсказания на обучающей части сохраняются "
            "начиная с этой версии, поэтому старые прогоны показывают только проверочную."
        )

    scores = [row.validation_score for row in folds if row.validation_score is not None]
    gaps = [row.gap for row in folds if row.gap is not None]
    mean_validation = round(float(np.mean(scores)), 6) if scores else None
    spread = round(float(np.std(scores, ddof=0)), 6) if len(scores) > 1 else None
    mean_gap = round(float(np.mean(gaps)), 6) if gaps else None

    findings = _findings(
        metric=chosen,
        folds=folds,
        mean_validation=mean_validation,
        spread=spread,
        mean_gap=mean_gap,
    )

    diagnostics = Diagnostics(
        contender_key=contender_key,
        task_type=task_type,
        metric=chosen,
        metric_label=label_of(chosen),
        higher_is_better=higher,
        folds=folds,
        mean_validation=mean_validation,
        spread_validation=spread,
        mean_gap=mean_gap,
        findings=findings,
        thresholds=thresholds.describe(),
        notes=notes,
    )

    if task_type == "regression":
        from backend.diagnostics.regression import regression_diagnostics

        diagnostics.regression = regression_diagnostics(validation)
    else:
        from backend.diagnostics.classification import classification_diagnostics

        diagnostics.classification = classification_diagnostics(
            validation, class_labels=class_labels, positive_label=positive_label
        )
        findings.extend(_imbalance_findings(diagnostics.classification))

    diagnostics.findings = sorted(
        findings, key=lambda item: (SEVERITY_ORDER.get(item.severity, 3), item.code)
    )
    return diagnostics


def _score_of(
    part: pl.DataFrame,
    *,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
    metric: str,
) -> float | None:
    if part.height == 0:
        return None

    #считается тем же модулем метрик, что и всё остальное: две реализации одной метрики
    #дают расхождение того же порядка, что и разрыв, который мы измеряем
    marked = part.with_columns(pl.lit(SPLIT_CV).alias("split"))
    sets: list[MetricSet] = evaluate(
        marked, task_type=task_type, class_labels=class_labels, positive_label=positive_label
    )

    if not sets:
        return None

    value = sets[0].by_key(metric)
    return value.value if value else None


def _fold_rows(
    *,
    validation: pl.DataFrame,
    probe: pl.DataFrame,
    task_type: str,
    class_labels: list[str],
    positive_label: str | None,
    metric: str,
    higher_is_better: bool,
) -> list[FoldRow]:
    rows: list[FoldRow] = []

    for fold in sorted(validation["fold"].unique().to_list()):
        fold_validation = validation.filter(pl.col("fold") == fold)
        fold_probe = probe.filter(pl.col("fold") == fold) if probe.height else probe
        shared = {
            "task_type": task_type,
            "class_labels": class_labels,
            "positive_label": positive_label,
            "metric": metric,
        }
        validation_score = _score_of(fold_validation, **shared)
        train_score = _score_of(fold_probe, **shared) if fold_probe.height else None
        gap = None

        if validation_score is not None and train_score is not None:
            #разрыв всегда в ориентации «больше — хуже»: для метрик, где меньше лучше,
            #переобучение выглядит как МЕНЬШАЯ ошибка на обучающей части
            gap = round(
                (train_score - validation_score) if higher_is_better
                else (validation_score - train_score),
                6,
            )

        rows.append(
            FoldRow(
                fold=int(fold),
                validation_rows=fold_validation.height,
                validation_score=validation_score,
                probe_rows=fold_probe.height,
                train_score=train_score,
                gap=gap,
            )
        )

    return rows


def _findings(
    *,
    metric: str,
    folds: list[FoldRow],
    mean_validation: float | None,
    spread: float | None,
    mean_gap: float | None,
) -> list[Finding]:
    findings: list[Finding] = []
    label = label_of(metric)
    train_scores = [row.train_score for row in folds if row.train_score is not None]
    mean_train = float(np.mean(train_scores)) if train_scores else None

    if mean_gap is not None and mean_train is not None:
        severity = thresholds.gap_severity(metric=metric, gap=mean_gap, train_score=mean_train)

        if severity != "info":
            relative = abs(mean_gap / mean_train) if mean_train else None
            findings.append(
                Finding(
                    code="train_validation_gap",
                    severity=severity,
                    title=f"Разрыв между обучающей и проверочной частью по {label}",
                    evidence={
                        "metric": metric,
                        "train_mean": round(mean_train, 6),
                        "validation_mean": mean_validation,
                        "gap": mean_gap,
                        "relative_gap": round(relative, 6) if relative is not None else None,
                        "relative_threshold": (
                            thresholds.RELATIVE_GAP_HIGH
                            if severity == "high"
                            else thresholds.RELATIVE_GAP_CAUTION
                        ),
                        "absolute_threshold": (
                            thresholds.ABSOLUTE_GAP_HIGH
                            if severity == "high"
                            else thresholds.ABSOLUTE_GAP_CAUTION
                        ),
                        "per_fold_gap": [row.gap for row in folds],
                    },
                    explanation=(
                        f"На обучающей части {label} = {mean_train:.4f}, на проверочной "
                        f"{mean_validation:.4f}. Разрыв {mean_gap:.4f} — модель описывает "
                        f"обучающие строки заметно лучше, чем новые."
                    ),
                    consequence=(
                        "На новых данных результат будет ближе к проверочному числу, "
                        "а не к обучающему. Ожидание, построенное на обучающей части, "
                        "окажется завышенным."
                    ),
                    suggestion=(
                        "Упростить модель, усилить регуляризацию или добавить данных. "
                        "Сравните разрыв с другими участниками: если он велик у всех, "
                        "дело в данных, а не в модели."
                    ),
                )
            )

    if spread is not None and mean_validation is not None:
        severity = thresholds.spread_severity(mean=mean_validation, spread=spread)

        if severity != "info":
            findings.append(
                Finding(
                    code="fold_instability",
                    severity=severity,
                    title=f"Результат сильно колеблется между фолдами по {label}",
                    evidence={
                        "metric": metric,
                        "mean": mean_validation,
                        "spread": spread,
                        "relative_spread": round(abs(spread / mean_validation), 6),
                        "threshold": (
                            thresholds.FOLD_SPREAD_HIGH
                            if severity == "high"
                            else thresholds.FOLD_SPREAD_CAUTION
                        ),
                        "per_fold": [row.validation_score for row in folds],
                    },
                    explanation=(
                        f"{label} по фолдам: "
                        + ", ".join(
                            f"{row.validation_score:.4f}"
                            for row in folds
                            if row.validation_score is not None
                        )
                        + f". Разброс {spread:.4f} при среднем {mean_validation:.4f}."
                    ),
                    consequence=(
                        "Одно среднее число скрывает несопоставимые фолды. Сравнение "
                        "с другой моделью по среднему при таком разбросе ненадёжно."
                    ),
                    suggestion=(
                        "Посмотрите, какой фолд выбивается: обычно дело в неоднородности "
                        "данных, а не в модели. Больше фолдов или больше данных сузят разброс."
                    ),
                )
            )

    missing = [row.fold for row in folds if row.validation_score is None]

    if missing:
        findings.append(
            Finding(
                code="metric_not_measurable",
                severity="caution",
                title=f"{label} посчитана не на всех фолдах",
                evidence={"folds_without_value": missing, "folds_total": len(folds)},
                explanation=(
                    f"На фолдах {missing} метрику посчитать не удалось — обычно потому, "
                    f"что на части присутствует один класс или модель не даёт вероятностей."
                ),
                consequence=(
                    "Среднее посчитано по неполному набору фолдов и не сопоставимо "
                    "напрямую со средним по всем."
                ),
                suggestion="Выберите метрику, определённую на этих данных, либо измените протокол.",
            )
        )

    return findings


def _imbalance_findings(classification: dict[str, Any] | None) -> list[Finding]:
    if not classification:
        return []

    findings: list[Finding] = []
    per_class = classification.get("per_class") or []
    total = sum(int(item.get("support", 0)) for item in per_class) or 1
    scarce = [item for item in per_class if int(item.get("support", 0)) < thresholds.MIN_CLASS_SUPPORT]

    if scarce:
        findings.append(
            Finding(
                code="scarce_class",
                severity="caution",
                title="У части классов слишком мало наблюдений для устойчивой оценки",
                evidence={
                    "classes": [
                        {"label": item["label"], "support": item["support"]} for item in scarce
                    ],
                    "threshold": thresholds.MIN_CLASS_SUPPORT,
                },
                explanation=(
                    "Классы "
                    + ", ".join(f"«{item['label']}» ({item['support']})" for item in scarce)
                    + f" встречаются реже {thresholds.MIN_CLASS_SUPPORT} раз."
                ),
                consequence=(
                    "Их precision и recall меняются на проценты от одной ошибки, "
                    "и сравнивать модели по этим числам нельзя."
                ),
                suggestion=(
                    "Объединить редкие классы, собрать больше наблюдений или читать "
                    "результат по классам с достаточным числом строк."
                ),
            )
        )

    rarest = min(per_class, key=lambda item: int(item.get("support", 0))) if per_class else None

    if rarest and int(rarest.get("support", 0)) / total < thresholds.SEVERE_IMBALANCE:
        share = int(rarest["support"]) / total
        findings.append(
            Finding(
                code="severe_imbalance",
                severity="caution",
                title="Сильный дисбаланс классов",
                evidence={
                    "rarest_label": rarest["label"],
                    "rarest_support": rarest["support"],
                    "share": round(share, 6),
                    "threshold": thresholds.SEVERE_IMBALANCE,
                },
                explanation=(
                    f"Самый редкий класс «{rarest['label']}» занимает {share:.1%} строк."
                ),
                consequence=(
                    "Accuracy при таком распределении почти не отличает модель от "
                    "постоянного ответа и вводит в заблуждение."
                ),
                suggestion=(
                    "Читайте результат по PR-AUC, recall редкого класса и balanced accuracy, "
                    "а не по accuracy."
                ),
            )
        )

    return findings
