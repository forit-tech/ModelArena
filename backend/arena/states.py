"""Состояния прогона и коды отказов.

Состояний намеренно больше, чем «получилось / не получилось». Разница между
«часть контендеров не обучилась», «пользователь остановил» и «backend перезапустили»
— это разные вопросы к результату, и склеивать их в один статус значит врать о том,
что произошло.

Главное правило слоя: **SUCCEEDED выдаётся только когда все запланированные контендеры
завершились успешно**. Прогон, где половина моделей упала, не имеет права выглядеть
как удачный: leaderboard из трёх строк вместо шести читается как «остальные проиграли».
"""
from __future__ import annotations

from typing import Literal

RunState = Literal[
    "PENDING",
    "RUNNING",
    "PARTIAL",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
]

ContenderState = Literal[
    "PENDING",
    "RUNNING",
    "SUCCEEDED",
    "SKIPPED",
    "FAILED",
    "CANCELLED",
]

TERMINAL_RUN_STATES: frozenset[str] = frozenset(
    {"PARTIAL", "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}
)
TERMINAL_CONTENDER_STATES: frozenset[str] = frozenset(
    {"SUCCEEDED", "SKIPPED", "FAILED", "CANCELLED"}
)

#коды стабильны и являются частью API: по ним фронт решает, что показать,
#и они не должны меняться вместе с текстом сообщения
ERROR_CODES: dict[str, str] = {
    "contender_unavailable": "Библиотека модели не установлена в этом окружении.",
    "incompatible_task": "Модель неприменима к этой задаче или к этим данным.",
    "preprocessing_failed": "Подготовка признаков завершилась ошибкой.",
    "training_failed": "Обучение модели завершилось ошибкой.",
    "prediction_failed": "Модель обучилась, но не смогла выдать предсказания.",
    "invalid_predictions": "Предсказания не прошли проверку формы или значений.",
    "metric_failed": "Метрики не удалось посчитать по сохранённым предсказаниям.",
    "timeout": "Превышен предел времени на контендера.",
    "resource_limit": "Превышен предел ресурсов.",
    "artifact_write_failed": "Не удалось записать артефакт прогона.",
    "worker_crashed": "Процесс контендера завершился аварийно.",
    "cancelled": "Остановлено пользователем.",
    "baseline_failed": "Baseline не обучился — под вопросом весь путь данных, а не одна модель.",
    "run_timeout": "Превышен предел времени на весь прогон.",
    "interrupted": "Backend был перезапущен во время прогона.",
}


def describe_error(code: str) -> str:
    return ERROR_CODES.get(code, "Неизвестная ошибка.")


def resolve_run_state(
    *,
    contender_states: list[str],
    cancelled: bool,
    baseline_failed: bool,
) -> RunState:
    """Свести состояния контендеров в состояние прогона.

    Порядок проверок задаёт приоритет причин:

    * упавший **baseline** перевешивает любые успехи. Если постоянный ответ не обучился,
      сломан общий путь данных, и остальные результаты нельзя считать осмысленными;
    * **отмена** перевешивает арифметику «сколько успело». Прогон неполон потому,
      что его остановили, а не потому, что модели проиграли, — и статус обязан
      называть настоящую причину;
    * `SUCCEEDED` требует, чтобы **не было** ни одного упавшего или отменённого.
      Пропущенный по применимости контендер (`SKIPPED`) в знаменатель неудач не входит:
      он был исключён до старта и с явной причиной.
    """
    if baseline_failed:
        return "FAILED"

    succeeded = [state for state in contender_states if state == "SUCCEEDED"]

    if cancelled:
        return "CANCELLED"

    failed = [state for state in contender_states if state in {"FAILED", "CANCELLED"}]

    if not succeeded:
        return "FAILED"

    if failed:
        return "PARTIAL"

    return "SUCCEEDED"
