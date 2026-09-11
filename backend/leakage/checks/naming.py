"""Подозрение по имени колонки.

Самый слабый из сигналов и единственный, который не измеряется на данных вовсе.
Поэтому его уровень — `heuristic`, и блокировать он не может никогда (D-23):
колонка `refund_reason` бывает и законным признаком, известным заранее.
"""
from __future__ import annotations

from backend.leakage.report import POST_OUTCOME_HINTS, LeakageSignal
from backend.tasks.spec import TaskSpec


def check(spec: TaskSpec) -> list[LeakageSignal]:
    suspicious = [
        name
        for name in spec.feature_columns
        if any(hint in name.lower() for hint in POST_OUTCOME_HINTS)
    ]

    if not suspicious:
        return []

    return [
        LeakageSignal(
            code="post_outcome_naming",
            evidence_level="heuristic",
            scope="features",
            columns=suspicious,
            observed_value=suspicious,
            threshold="совпадение с перечнем слов, обозначающих исход",
            explanation=(
                f"Имена {', '.join(suspicious)} намекают на событие, происходящее после того, "
                "как исход стал известен: причина, итог, отмена, возврат."
            ),
            mechanism=(
                "Если поле действительно заполняется постфактум, оно несёт ответ, и сравнение "
                "моделей выродится. Но это подозрение ТОЛЬКО по имени: содержимое не проверялось, "
                "и колонка вполне может быть законной."
            ),
            suggested_action=(
                "Сверьтесь с бизнес-процессом: доступно ли каждое из этих полей в момент "
                "предсказания. Если да — ничего делать не нужно."
            ),
        )
    ]
