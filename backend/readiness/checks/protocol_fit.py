"""Соответствие протокола структуре данных.

Самая дорогая ошибка сравнения — не плохие данные, а правильные данные, проверенные
неправильным способом. Временная колонка сама по себе не плоха; плохо игнорировать её
при случайном разбиении. Здесь проверяется именно связка, а не датасет.
"""
from __future__ import annotations

import polars as pl

from backend.protocol.recommend import MIN_ROWS_FOR_INTERVALS, ProtocolProposal
from backend.readiness.report import Finding, ReadinessContext
from backend.tasks.inference import find_group_column_candidates, find_time_column_candidates
from backend.tasks.spec import TaskSpec

TIME_AWARE_SPLITTERS = {"forward_chaining"}
GROUP_AWARE_SPLITTERS = {"group_k_fold", "stratified_group_k_fold"}


def check(
    frame: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
    context: ReadinessContext,
) -> list[Finding]:
    findings: list[Finding] = []
    features = frame.select([name for name in spec.feature_columns if name in frame.columns])

    findings.extend(_time_structure(features, spec, protocol))
    findings.extend(_group_structure(features, spec, protocol))
    findings.extend(_holdout_size(context))
    findings.extend(_reduced_folds(protocol))

    return findings


def _time_structure(
    features: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
) -> list[Finding]:
    if protocol.cv_splitter in TIME_AWARE_SPLITTERS or spec.time_column:
        #структура учтена — находки нет. Наличие даты само по себе не проблема
        return []

    candidates = find_time_column_candidates(features)

    if not candidates:
        return []

    return [
        Finding(
            code="time_structure_ignored",
            severity="high_risk",
            scope="protocol",
            observed_value=candidates[:3],
            threshold="при наличии временной структуры разбиение обязано быть по времени",
            explanation=(
                f"Среди признаков есть колонки со временем ({', '.join(candidates[:3])}), "
                f"а выбранное разбиение — «{protocol.cv_splitter}»."
            ),
            consequence=(
                "Часть обучающих строк окажется позже проверочных: модель будет знать будущее. "
                "Метрика окажется оптимистичной, причём сильнее у моделей, способных уловить "
                "временной тренд, — то есть leaderboard наградит именно за утечку."
            ),
            suggested_action=(
                "Выберите колонку времени в настройках протокола либо исключите эти колонки "
                "из признаков, если порядок наблюдений не имеет значения."
            ),
        )
    ]


def _group_structure(
    features: pl.DataFrame,
    spec: TaskSpec,
    protocol: ProtocolProposal,
) -> list[Finding]:
    if protocol.cv_splitter in GROUP_AWARE_SPLITTERS or spec.group_column:
        return []

    candidates = find_group_column_candidates(features)

    if not candidates:
        return []

    return [
        Finding(
            code="group_structure_ignored",
            severity="high_risk",
            scope="protocol",
            observed_value=candidates[:3],
            threshold="повторяющаяся сущность требует разбиения по группам",
            explanation=(
                f"Среди признаков есть колонки, похожие на идентификатор повторяющейся сущности "
                f"({', '.join(candidates[:3])}), а разбиение — «{protocol.cv_splitter}»."
            ),
            consequence=(
                "Строки одной сущности разойдутся между обучением и проверкой, и модель будет "
                "узнавать сущность, а не закономерность. Завышение достанется прежде всего "
                "моделям с большой ёмкостью — сравнение станет нечестным именно там, "
                "где оно интереснее всего."
            ),
            suggested_action=(
                "Выберите колонку группировки в настройках протокола. Если повторов на самом "
                "деле нет — ничего делать не нужно, это подозрение по форме данных."
            ),
        )
    ]


def _holdout_size(context: ReadinessContext) -> list[Finding]:
    if context.holdout_rows >= MIN_ROWS_FOR_INTERVALS:
        return []

    return [
        Finding(
            code="holdout_too_small_for_intervals",
            severity="caution",
            scope="protocol",
            observed_value=context.holdout_rows,
            threshold=f"{MIN_ROWS_FOR_INTERVALS} строк для доверительного интервала",
            explanation=f"В holdout попадёт около {context.holdout_rows} строк.",
            consequence=(
                "Доверительный интервал на такой выборке шире самой метрики, поэтому "
                "подтверждение чемпиона по holdout будет чисто формальным."
            ),
            suggested_action=(
                "Ничего делать не нужно, но решение принимайте по кросс-валидации, "
                "а holdout читайте как грубую проверку на отсутствие грубых ошибок."
            ),
        )
    ]


def _reduced_folds(protocol: ProtocolProposal) -> list[Finding]:
    if not protocol.feasibility_notes:
        return []

    return [
        Finding(
            code="protocol_adjusted",
            severity="info",
            scope="protocol",
            observed_value=protocol.n_splits,
            threshold="запрошенное число фолдов",
            explanation="; ".join(protocol.feasibility_notes),
            consequence=(
                "Протокол подстроен под данные. Само по себе это не искажает сравнение, "
                "но объясняет, почему в результатах другое число фолдов, чем в настройках."
            ),
            suggested_action="Ничего делать не нужно — достаточно знать об этом при чтении результатов.",
        )
    ]
