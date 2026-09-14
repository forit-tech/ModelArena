"""Артефакт модели: обученный конвейер плюс всё, без чего он бесполезен.

Голый `pickle` с обученной моделью — не артефакт, а ловушка. Через месяц по нему нельзя
ответить ни на один нужный вопрос: какие колонки он ждёт и в каком порядке, что означает
столбец вероятностей, на каком датасете он обучен, той же ли версией библиотеки его
загружают. Ошибка в любом из этих пунктов даёт не отказ, а **правдоподобные неверные
предсказания**.

Поэтому рядом с моделью лежит манифест, а сама загрузка проверяет:

* формат и мажорную версию — чужой или более новый артефакт отвергается, а не угадывается;
* контрольную сумму — повреждённый файл не должен молча загрузиться наполовину;
* версии библиотек — расхождение показывается предупреждением, потому что sklearn
  не обещает совместимости сериализации между версиями.

**Граница доверия.** `joblib` при загрузке **исполняет код**. Поэтому загружаются только
артефакты, созданные этим же приложением и лежащие внутри его каталога прогонов, а их
целостность сверяется с суммой, записанной в карточке прогона. Артефакт, скачанный
из интернета, этим слоем не поддерживается — и обещать обратное было бы опасной неправдой.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "model.json"
ARTIFACT_FORMAT = "modelarena.model"
#мажорная версия меняется, когда старый загрузчик не сможет прочитать новый артефакт
ARTIFACT_VERSION = "1.0"
CHUNK = 1024 * 1024


class ArtifactError(Exception):
    """Артефакт отвергнут. Несёт машинный код для ответа API."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FeatureSchema:
    """Что именно конвейер ждёт на входе.

    Порядок колонок хранится явно: `ColumnTransformer` обращается к ним по имени,
    но пользовательский файл может прийти с другим порядком, лишними или недостающими
    колонками, и разбираться с этим нужно до предсказания, а не после.
    """

    columns: list[str]
    dtypes: dict[str, str]
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "dtypes": self.dtypes,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "datetime": self.datetime,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSchema:
        return cls(
            columns=list(payload.get("columns", [])),
            dtypes=dict(payload.get("dtypes", {})),
            numeric=list(payload.get("numeric", [])),
            categorical=list(payload.get("categorical", [])),
            datetime=list(payload.get("datetime", [])),
        )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)

    return digest.hexdigest()


def environment() -> dict[str, str]:
    """Версии, от которых зависит возможность прочитать артефакт обратно.

    scikit-learn не обещает совместимости сериализации между версиями и сам предупреждает
    об этом при загрузке. Записываем версии, чтобы расхождение можно было **назвать**,
    а не обнаружить по странным предсказаниям.
    """
    versions = {"python": platform.python_version(), "platform": platform.platform()}

    for name in ("sklearn", "numpy", "pandas", "polars", "joblib"):
        try:
            module = __import__(name)
        except ImportError:
            continue

        versions[name] = str(getattr(module, "__version__", "неизвестно"))

    return versions


def build_manifest(
    *,
    run_id: str,
    contender_key: str,
    adapter_key: str,
    label: str,
    params: dict[str, Any],
    preprocessing_profile: str,
    task: dict[str, Any],
    dataset: dict[str, Any],
    protocol: dict[str, Any],
    schema: FeatureSchema,
    training_rows: int,
    created_at: str,
) -> dict[str, Any]:
    """Собрать манифест. Контрольная сумма добавляется после записи файла модели."""
    return {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "created_at": created_at,
        "run_id": run_id,
        "contender_key": contender_key,
        "adapter_key": adapter_key,
        "label": label,
        "params": params,
        "preprocessing_profile": preprocessing_profile,
        #постановка задачи целиком: без неё непонятно, что означает выход модели
        "task": task,
        #к какому именно снимку данных относится модель
        "dataset": dataset,
        "protocol": protocol,
        "feature_schema": schema.to_dict(),
        "training_rows": training_rows,
        "environment": environment(),
        "model_sha256": "",
    }


def write_artifact(directory: Path, pipeline: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """Записать модель и манифест. Возвращает манифест с проставленной суммой."""
    import joblib

    model_path = directory / MODEL_FILE
    joblib.dump(pipeline, model_path, compress=3)
    manifest = dict(manifest)
    manifest["model_sha256"] = sha256_of(model_path)
    manifest["model_bytes"] = model_path.stat().st_size
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_FILE

    if not path.exists():
        raise ArtifactError(
            "artifact_missing", "Манифест модели не найден: артефакт неполон или не создавался."
        )

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError("artifact_corrupted", "Манифест модели не читается как JSON.") from error

    if not isinstance(manifest, dict):
        raise ArtifactError("artifact_corrupted", "Манифест модели имеет неожиданную структуру.")

    return manifest


def verify(directory: Path) -> dict[str, Any]:
    """Проверить артефакт, ничего не загружая и не исполняя.

    Сначала проверка, потом загрузка — и никогда наоборот: `joblib.load` исполняет код,
    поэтому отказ должен случаться **до** него.
    """
    manifest = read_manifest(directory)

    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError(
            "artifact_foreign",
            f"Ожидался формат {ARTIFACT_FORMAT}, получен «{manifest.get('format')}». "
            "Чужие артефакты этим слоем не загружаются.",
        )

    declared = str(manifest.get("format_version", ""))
    major = declared.split(".", maxsplit=1)[0]

    if major != ARTIFACT_VERSION.split(".", maxsplit=1)[0]:
        raise ArtifactError(
            "artifact_incompatible",
            f"Мажорная версия формата артефакта {declared} не поддерживается "
            f"(эта сборка читает {ARTIFACT_VERSION}). Угадывать структуру нельзя.",
        )

    model_path = directory / MODEL_FILE

    if not model_path.exists():
        raise ArtifactError("artifact_missing", "Файл модели отсутствует рядом с манифестом.")

    expected = str(manifest.get("model_sha256", ""))

    if not expected:
        raise ArtifactError(
            "artifact_corrupted", "В манифесте нет контрольной суммы модели: проверить нечем."
        )

    actual = sha256_of(model_path)

    if actual != expected:
        raise ArtifactError(
            "artifact_corrupted",
            "Контрольная сумма файла модели не совпадает с записанной в манифесте: "
            "файл повреждён или подменён.",
        )

    return manifest


def environment_warnings(manifest: dict[str, Any]) -> list[str]:
    """Расхождения версий между записью и чтением артефакта.

    Это предупреждение, а не отказ: артефакт часто читается той же средой и работает.
    Но молчать нельзя — sklearn не гарантирует совместимости сериализации между версиями.
    """
    recorded = dict(manifest.get("environment") or {})
    current = environment()
    notes: list[str] = []

    for name in ("sklearn", "numpy", "python"):
        was, now = recorded.get(name), current.get(name)

        if was and now and was != now:
            notes.append(
                f"{name}: артефакт записан версией {was}, читается версией {now}. "
                "Совместимость сериализации между версиями не гарантируется."
            )

    return notes


def load_artifact(directory: Path) -> tuple[Any, dict[str, Any]]:
    """Проверить и загрузить обученный конвейер.

    Загружается **только** артефакт, созданный этим приложением: `joblib` исполняет код
    при десериализации, и открывать этим путём произвольные файлы нельзя.
    """
    manifest = verify(directory)

    import joblib

    try:
        pipeline = joblib.load(directory / MODEL_FILE)
    except Exception as error:
        #сюда попадает и несовместимость версий, и повреждение, пережившее сверку суммы
        raise ArtifactError(
            "artifact_unreadable",
            f"Модель не загружается: {type(error).__name__}. "
            "Возможная причина — несовместимая версия библиотеки.",
        ) from error

    if not hasattr(pipeline, "predict"):
        raise ArtifactError(
            "artifact_corrupted", "В файле модели оказался объект, который не умеет предсказывать."
        )

    return pipeline, manifest
