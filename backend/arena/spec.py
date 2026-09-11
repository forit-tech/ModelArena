"""Неизменяемое описание прогона.

`RunSpec` фиксируется в момент старта и больше не меняется. Это не формальность:
карточка прогона отвечает на вопрос «что именно тут сравнивали» через месяцы, и если
входы можно поправить задним числом, ответ становится недостоверным.

Отпечаток считается **по эксперименту, а не по машине**. Число потоков и предел памяти
в него не входят: тот же датасет, та же задача, тот же протокол и те же участники дают
тот же эксперимент независимо от того, на скольких ядрах его прогнали. Иначе один и тот же
прогон, запущенный на ноутбуке и на сервере, выглядел бы как два разных, а защита
от случайного двойного запуска перестала бы работать.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.models.context import ResourceBudget
from backend.models.registry import Contender
from backend.protocol.folds import FoldPlan
from backend.protocol.recommend import ProtocolProposal
from backend.tasks.spec import TaskSpec

#предел времени на одного контендера: без него зависшая нативная библиотека держит
#прогон бесконечно, а пользователь видит RUNNING, за которым ничего не происходит
DEFAULT_CONTENDER_TIMEOUT_SECONDS = 1800
DEFAULT_RUN_TIMEOUT_SECONDS = 7200


@dataclass(frozen=True)
class RunSpec:
    """Входы прогона. После старта неизменяемы."""

    dataset_id: str
    dataset_fingerprint: str
    task: TaskSpec
    protocol: ProtocolProposal
    fold_assignment_hash: str
    n_splits: int
    holdout_rows: int
    train_pool_rows: int
    contenders: list[Contender]
    budget: ResourceBudget
    seed: int
    contender_timeout_seconds: int = DEFAULT_CONTENDER_TIMEOUT_SECONDS
    run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS
    #считается в __post_init__ и участвует в защите от случайного двойного запуска
    experiment_fingerprint: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.experiment_fingerprint:
            object.__setattr__(self, "experiment_fingerprint", self._fingerprint())

    @property
    def runnable_contenders(self) -> list[Contender]:
        return [contender for contender in self.contenders if contender.runnable]

    @property
    def baseline_key(self) -> str | None:
        return next(
            (contender.contender_key for contender in self.contenders if contender.is_baseline),
            None,
        )

    def _fingerprint(self) -> str:
        #в отпечаток входит только то, что определяет сам эксперимент. Порядок ключей
        #фиксирован сортировкой: словарь с теми же полями в другом порядке — тот же прогон
        payload = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "task": self.task.to_dict(),
            "protocol": self.protocol.to_dict(),
            #хеш разбиения, а не параметры протокола: два протокола с одинаковыми
            #параметрами, но разными фактическими фолдами — разные эксперименты
            "fold_assignment_hash": self.fold_assignment_hash,
            "seed": self.seed,
            "contenders": sorted(
                #параметры входят в отпечаток: та же модель с другой глубиной — другой участник
                (contender.contender_key, json.dumps(contender.params, sort_keys=True, default=str))
                for contender in self.contenders
                if contender.runnable
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "task": self.task.to_dict(),
            "protocol": self.protocol.to_dict(),
            "fold_assignment_hash": self.fold_assignment_hash,
            "n_splits": self.n_splits,
            "holdout_rows": self.holdout_rows,
            "train_pool_rows": self.train_pool_rows,
            "contenders": [contender.to_dict() for contender in self.contenders],
            "budget": asdict(self.budget),
            "seed": self.seed,
            "contender_timeout_seconds": self.contender_timeout_seconds,
            "run_timeout_seconds": self.run_timeout_seconds,
            "experiment_fingerprint": self.experiment_fingerprint,
        }


def build_run_spec(
    *,
    dataset_id: str,
    dataset_fingerprint: str,
    task: TaskSpec,
    protocol: ProtocolProposal,
    folds: FoldPlan,
    contenders: list[Contender],
    budget: ResourceBudget,
    contender_timeout_seconds: int = DEFAULT_CONTENDER_TIMEOUT_SECONDS,
    run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS,
) -> RunSpec:
    """Собрать спецификацию прогона из уже утверждённых частей.

    Seed берётся из протокола, а не задаётся отдельно: два источника случайности
    рано или поздно разошлись бы, и «тот же прогон» перестал бы воспроизводиться.
    """
    return RunSpec(
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        task=task,
        protocol=protocol,
        fold_assignment_hash=folds.assignment_hash,
        n_splits=folds.n_splits,
        holdout_rows=len(folds.holdout),
        train_pool_rows=len(folds.train_pool),
        contenders=list(contenders),
        budget=budget,
        seed=protocol.seed,
        contender_timeout_seconds=contender_timeout_seconds,
        run_timeout_seconds=run_timeout_seconds,
    )
