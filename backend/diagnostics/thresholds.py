"""Пороги диагностики — в одном месте и наружу, а не внутри интерфейса.

Порог, зашитый в компонент экрана, невозможно ни обсудить, ни воспроизвести: пользователь
видит слово «переобучение» и не может узнать, при каком числе оно появляется. Поэтому все
пороги объявлены здесь, объяснены и отдаются вместе с находкой.

Ни один из них не является законом природы. Это рабочие ориентиры, и каждая находка несёт
измеренные числа рядом с порогом — чтобы решение принимал человек, а не константа.
"""
from __future__ import annotations

from typing import Any

#разрыв между обучающей и проверочной частью в долях самой метрики. Относительная
#величина нужна потому, что абсолютные 0.05 значат разное для ROC-AUC и для RMSE
RELATIVE_GAP_CAUTION = 0.10
RELATIVE_GAP_HIGH = 0.25

#для метрик, ограниченных отрезком [0, 1], разумен и абсолютный порог: относительная
#величина при близком к нулю знаменателе взрывается и перестаёт что-либо означать
ABSOLUTE_GAP_CAUTION = 0.05
ABSOLUTE_GAP_HIGH = 0.15

#разброс результата между фолдами в долях среднего. Большой разброс означает, что
#одно число «средняя метрика» скрывает несопоставимые фолды
FOLD_SPREAD_CAUTION = 0.10
FOLD_SPREAD_HIGH = 0.25

#класс, которому досталось меньше этого числа строк, не даёт устойчивой оценки:
#одна ошибка меняет его метрику на проценты
MIN_CLASS_SUPPORT = 20

#доля самого редкого класса, ниже которой accuracy перестаёт быть информативной
SEVERE_IMBALANCE = 0.05

#сколько ошибочных строк показывать в разборе по умолчанию
ERROR_ROWS_DEFAULT = 50
ERROR_ROWS_MAX = 500

#метрики, значения которых лежат в [0, 1] и допускают абсолютный порог
BOUNDED_METRICS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "macro_f1",
        "brier",
        "r2",
    }
)


def describe() -> dict[str, Any]:
    """Пороги в виде, пригодном для показа пользователю."""
    return {
        "relative_gap": {
            "caution": RELATIVE_GAP_CAUTION,
            "high": RELATIVE_GAP_HIGH,
            "meaning": (
                "Разрыв между обучающей и проверочной частью в долях метрики. "
                "Относительная величина, потому что абсолютные 0.05 значат разное "
                "для ROC-AUC и для RMSE."
            ),
        },
        "absolute_gap": {
            "caution": ABSOLUTE_GAP_CAUTION,
            "high": ABSOLUTE_GAP_HIGH,
            "meaning": (
                "Применяется к метрикам на отрезке [0, 1]: относительная величина "
                "при знаменателе около нуля перестаёт что-либо означать."
            ),
        },
        "fold_spread": {
            "caution": FOLD_SPREAD_CAUTION,
            "high": FOLD_SPREAD_HIGH,
            "meaning": (
                "Разброс результата между фолдами в долях среднего. Большой разброс "
                "означает, что одно среднее скрывает несопоставимые фолды."
            ),
        },
        "min_class_support": {
            "value": MIN_CLASS_SUPPORT,
            "meaning": (
                "Класс с меньшим числом наблюдений не даёт устойчивой оценки: "
                "одна ошибка меняет его метрику на проценты."
            ),
        },
        "severe_imbalance": {
            "value": SEVERE_IMBALANCE,
            "meaning": (
                "Доля самого редкого класса, ниже которой accuracy перестаёт быть "
                "информативной: постоянный ответ даёт почти такое же число."
            ),
        },
    }


def gap_severity(*, metric: str, gap: float, train_score: float) -> str:
    """Насколько серьёзен разрыв train-vs-validation.

    Возвращает `info`, `caution` или `high`. Отрицательный разрыв (проверочная часть
    лучше обучающей) переобучением не является и здесь не отмечается: это обычно
    следствие регуляризации или удачного разбиения, а не признак беды.
    """
    if gap <= 0:
        return "info"

    relative = abs(gap / train_score) if train_score else float("inf")
    bounded = metric in BOUNDED_METRICS

    if relative >= RELATIVE_GAP_HIGH or (bounded and gap >= ABSOLUTE_GAP_HIGH):
        return "high"

    if relative >= RELATIVE_GAP_CAUTION or (bounded and gap >= ABSOLUTE_GAP_CAUTION):
        return "caution"

    return "info"


def spread_severity(*, mean: float, spread: float) -> str:
    if not mean:
        return "info"

    relative = abs(spread / mean)

    if relative >= FOLD_SPREAD_HIGH:
        return "high"

    if relative >= FOLD_SPREAD_CAUTION:
        return "caution"

    return "info"
