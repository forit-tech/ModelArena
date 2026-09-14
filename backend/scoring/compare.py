"""Парное сравнение двух моделей по фолдам.

Сравнивать средние с их разбросами по отдельности нельзя: разброс между фолдами берётся
в основном от самих фолдов, а не от модели. Один фолд оказался труднее — просядут **все**
участники сразу, и «широкий разброс» ошибочно прочитается как неустойчивость модели.

Здесь считается другое: разница **на каждом фолде в отдельности**. Это законно ровно
потому, что все контендеры обучались на одном и том же разбиении (D-7): общая трудность
фолда входит в оба слагаемых и при вычитании уходит. Остаётся то, чем модели отличаются.

**Это не статистический тест.** Пять фолдов одного датасета не являются независимыми
наблюдениями: их обучающие части пересекаются между собой. p-value здесь было бы
выдуманной точностью. Поэтому слой сообщает наблюдаемую картину — на скольких фолдах
одна модель впереди и насколько разница велика по сравнению с её собственным колебанием, —
и не притворяется, что проверил гипотезу.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#разница считается устойчивой, если она не меньше собственного колебания по фолдам.
#Величина выбрана как «сигнал не тонет в шуме», а не как уровень значимости
STABILITY_RATIO = 1.0


@dataclass(frozen=True)
class PairedComparison:
    """Чем одна модель отличается от другой на общих фолдах."""

    leader: str
    trailing: str
    metric: str
    #положительная величина означает превосходство leader, знак метрики уже учтён
    mean_difference: float
    std_difference: float
    folds_compared: int
    folds_won: int
    stable: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "leader": self.leader,
            "trailing": self.trailing,
            "metric": self.metric,
            "mean_difference": self.mean_difference,
            "std_difference": self.std_difference,
            "folds_compared": self.folds_compared,
            "folds_won": self.folds_won,
            "stable": self.stable,
            "explanation": self.explanation,
        }


def compare_per_fold(
    *,
    leader_key: str,
    leader_folds: list[float | None],
    trailing_key: str,
    trailing_folds: list[float | None],
    metric: str,
    higher_is_better: bool,
) -> PairedComparison | None:
    """Сравнить две модели по значениям метрики на одинаковых фолдах.

    `None` возвращается, если сравнивать не на чем: у одной из моделей метрика
    не посчиталась ни на одном общем фолде. Это отдельный случай, а не «ничья».
    """
    pairs = [
        (float(first), float(second))
        for first, second in zip(leader_folds, trailing_folds, strict=False)
        if first is not None and second is not None
    ]

    if not pairs:
        return None

    #разница берётся пофолдно: общая трудность фолда входит в оба значения и сокращается
    sign = 1.0 if higher_is_better else -1.0
    differences = np.array([sign * (first - second) for first, second in pairs], dtype=np.float64)
    mean = float(differences.mean())
    spread = float(differences.std(ddof=0))
    won = int((differences > 0).sum())
    #устойчивой считается разница, которая сохраняет знак на всех фолдах и превышает
    #собственное колебание. Одного среднего мало: выигрыш на четырёх фолдах и крупный
    #проигрыш на пятом даёт положительное среднее при неустойчивом превосходстве
    consistent = won == len(differences) or won == 0
    stable = bool(consistent and abs(mean) > STABILITY_RATIO * spread)

    return PairedComparison(
        leader=leader_key,
        trailing=trailing_key,
        metric=metric,
        mean_difference=round(mean, 6),
        std_difference=round(spread, 6),
        folds_compared=len(differences),
        folds_won=won,
        stable=stable,
        explanation=_explain(
            metric=metric,
            mean=mean,
            spread=spread,
            won=won,
            total=len(differences),
            stable=stable,
        ),
    )


def _explain(
    *, metric: str, mean: float, spread: float, won: int, total: int, stable: bool
) -> str:
    if stable and mean > 0:
        return (
            f"Впереди на всех {total} фолдах, {metric} выше в среднем на {mean:.4f} "
            f"при колебании разницы {spread:.4f} — превосходство держится на каждом фолде."
        )

    if stable and mean < 0:
        return (
            f"Отстаёт на всех {total} фолдах, {metric} ниже в среднем на {abs(mean):.4f} "
            f"при колебании разницы {spread:.4f}."
        )

    if abs(mean) <= STABILITY_RATIO * spread:
        return (
            f"Разница по {metric} составляет {mean:+.4f} при колебании {spread:.4f} — "
            f"она меньше собственного разброса между фолдами, то есть модели неразличимы. "
            f"Выигрыш на {won} фолдах из {total}."
        )

    return (
        f"Разница по {metric} составляет {mean:+.4f}, но знак меняется между фолдами "
        f"(выигрыш на {won} из {total}): превосходство не держится на всём разбиении."
    )
