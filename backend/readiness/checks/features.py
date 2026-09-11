"""Проверки признаков — **только выбранных**.

Контекстность здесь принципиальна (D-22). 30% пропусков в колонке, которую пользователь
не взял в признаки, не влияют ни на что. `customer_id`, оставленный признаком, — риск;
он же, выбранный колонкой группировки, — полезная структура. Проверять датасет целиком
и пугать находками по неиспользуемым колонкам значит приучить их пролистывать.
"""
from __future__ import annotations

import polars as pl

from backend.datasets.profile import build_profile
from backend.datasets.registry import DatasetSnapshot
from backend.readiness.report import (
    DUPLICATE_HIGH_RISK,
    MISSING_CAUTION,
    MISSING_HIGH_RISK,
    ONE_HOT_CAP,
    Finding,
    ReadinessContext,
)
from backend.tasks.spec import TaskSpec


def check(
    snapshot: DatasetSnapshot,  # noqa: ARG001
    frame: pl.DataFrame,
    spec: TaskSpec,
    context: ReadinessContext,
) -> list[Finding]:
    selected = set(spec.feature_columns)
    profile = build_profile(frame.select([*spec.feature_columns, spec.target_column]))
    findings: list[Finding] = []

    for column in profile.columns:
        if column.name not in selected:
            continue

        findings.extend(_column_findings(column, spec, context))

    findings.extend(_duplicates(profile.duplicate_row_count, context))
    return findings


def _column_findings(column, spec: TaskSpec, context: ReadinessContext) -> list[Finding]:
    findings: list[Finding] = []

    if column.is_probable_id:
        #та же колонка в роли группировки — не проблема, а структура, которую надо уважать
        role = (
            "выбрана колонкой группировки"
            if column.name == spec.group_column
            else "оставлена признаком"
        )

        if column.name != spec.group_column:
            findings.append(
                Finding(
                    code="identifier_used_as_feature",
                    severity="high_risk",
                    scope=f"column:{column.name}",
                    observed_value=round(column.unique_ratio, 4),
                    threshold="уникальность признака ниже 0.98",
                    explanation=(
                        f"«{column.name}» почти уникальна ({column.unique_ratio:.0%} различных "
                        f"значений) и {role}. {column.id_reason or ''}".strip()
                    ),
                    consequence=(
                        "Модель может запомнить конкретные значения вместо закономерности: "
                        "на кросс-валидации это даёт завышенную метрику, а на новых данных "
                        "таких значений не будет вовсе. Сравнение моделей превратится "
                        "в сравнение способности запоминать."
                    ),
                    suggested_action=(
                        "Исключите колонку из признаков. Если строки относятся к повторяющимся "
                        "сущностям — выберите её колонкой группировки: тогда она станет "
                        "защитой, а не риском."
                    ),
                )
            )

    if column.is_constant:
        findings.append(
            Finding(
                code="constant_feature",
                severity="caution",
                scope=f"column:{column.name}",
                observed_value=column.unique_count,
                threshold="хотя бы два различных значения",
                explanation=f"«{column.name}» принимает одно значение на всю выборку.",
                consequence=(
                    "Признак не различает объекты и не влияет на предсказание, "
                    "но занимает место в пайплайне и в отчёте о важности."
                ),
                suggested_action="Уберите из признаков — на сравнение это не повлияет.",
            )
        )

    if column.missing_ratio >= MISSING_HIGH_RISK:
        findings.append(
            Finding(
                code="feature_mostly_missing",
                severity="high_risk",
                scope=f"column:{column.name}",
                observed_value=round(column.missing_ratio, 4),
                threshold=f"{MISSING_CAUTION:.0%} — внимание, {MISSING_HIGH_RISK:.0%} — риск",
                explanation=f"В признаке «{column.name}» {column.missing_ratio:.0%} пропусков.",
                consequence=(
                    "Заполнение почти пустой колонки создаёт признак из константы-заглушки. "
                    "Модели, по-разному чувствительные к заполнению, окажутся несравнимы: "
                    "разница между ними будет отражать реакцию на заглушку, а не на данные."
                ),
                suggested_action="Исключите колонку или замените признаком «значение известно».",
            )
        )
    elif column.missing_ratio >= MISSING_CAUTION:
        findings.append(
            Finding(
                code="feature_missing_values",
                severity="caution",
                scope=f"column:{column.name}",
                observed_value=round(column.missing_ratio, 4),
                threshold=f"{MISSING_CAUTION:.0%} пропусков",
                explanation=f"В признаке «{column.name}» {column.missing_ratio:.0%} пропусков.",
                consequence=(
                    "Способ заполнения станет частью различий между моделями разных семейств: "
                    "часть из них работает с пропусками сама, часть получает медиану."
                ),
                suggested_action=(
                    "Ничего делать не обязательно, но при чтении leaderboard помните: "
                    "здесь сравниваются и стратегии заполнения тоже."
                ),
            )
        )

    if not column.is_probable_id and column.unique_count > ONE_HOT_CAP and column.logical_type == "string":
        findings.append(
            Finding(
                code="high_cardinality_feature",
                severity="caution",
                scope=f"column:{column.name}",
                observed_value=column.unique_count,
                threshold=f"{ONE_HOT_CAP} уровней — предел one-hot",
                explanation=(
                    f"«{column.name}» содержит {column.unique_count} различных значений "
                    f"при {context.usable_rows} строках."
                ),
                consequence=(
                    "После кодирования редкие уровни схлопнутся в общий, и модели, "
                    "которые умеют работать с категориями сами, получат другую информацию, "
                    "чем модели на one-hot. Это делает их сравнение менее прямым."
                ),
                suggested_action="Объедините редкие уровни в DataArena или оставьте как есть, учитывая оговорку.",
            )
        )

    return findings


def _duplicates(duplicate_rows: int, context: ReadinessContext) -> list[Finding]:
    if duplicate_rows == 0:
        return []

    ratio = duplicate_rows / context.total_rows
    severity = "high_risk" if ratio >= DUPLICATE_HIGH_RISK else "caution"

    return [
        Finding(
            code="duplicate_rows",
            severity=severity,
            scope="dataset",
            observed_value=duplicate_rows,
            threshold=f"{DUPLICATE_HIGH_RISK:.0%} доли полных дублей",
            explanation=f"{duplicate_rows} полностью одинаковых строк ({ratio:.1%}).",
            consequence=(
                "Копии одной строки расходятся между обучением и проверкой, и модель отвечает "
                "на вопрос, который уже видела с ответом. Метрика измеряет память, а не "
                "обобщение, причём у гибких моделей сильнее — то есть leaderboard смещается."
            ),
            suggested_action=(
                "Удалите дубли в DataArena либо выберите разбиение по сущности, "
                "если повторы осмысленны."
            ),
        )
    ]
