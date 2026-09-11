"""Реестр неизменяемых снимков датасетов.

Снимок — то, на чём проводится эксперимент. Он **неизменяем**: любая правка данных даёт
новый снимок со ссылкой на родителя. Именно это делает эксперименты воспроизводимыми:
карточка прогона ссылается на `dataset_id`, и содержимое под ним не может поменяться
задним числом.

Идентичность снимка — логический отпечаток `dataarena-logical-sha256-v1` (D-14, D-16),
тот же, что в контракте. Поэтому один и тот же датасет, пришедший пакетом и файлом,
опознаётся как один снимок, а не задваивается.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from backend.core.config import get_settings
from backend.core.errors import NotFoundError, ValidationError
from backend.datasets.io import read_table
from backend.datasets.package import content_fingerprint, read_package

logger = logging.getLogger("modelarena.datasets")

SNAPSHOT_FILE = "data.parquet"
META_FILE = "meta.json"
#служебный идентификатор строки: нужен, чтобы Head-to-Head и разбор ошибок могли показать
#конкретные объекты. Признаком не становится никогда (инвариант I-8)
ROW_ID = "__row_id__"
DATASET_ID_PATTERN = re.compile(r"^ds_[0-9a-f]{16,48}$")
MIN_ROWS = 2
MIN_COLUMNS = 2


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    logical_type: str
    nullable: bool


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str
    name: str
    source: Literal["package", "file"]
    fingerprint: str
    fingerprint_algorithm: str
    row_count: int
    column_count: int
    columns: list[ColumnSpec]
    created_at: str
    row_key: list[str] = field(default_factory=list)
    package_ref: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetSnapshot:
        #неизвестные поля игнорируются: карточка, записанная более новой версией,
        #не должна ронять список датасетов
        known = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in payload.items() if key in known}
        filtered["columns"] = [ColumnSpec(**column) for column in payload.get("columns", [])]
        return cls(**filtered)


def _new_dataset_id() -> str:
    #сортируемый по времени идентификатор без внешней зависимости: миллисекунды + случайность
    return f"ds_{int(time.time() * 1000):011x}{secrets.token_hex(5)}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _describe_columns(frame: pl.DataFrame) -> list[ColumnSpec]:
    from backend.datasets.package import logical_type

    return [
        ColumnSpec(
            name=name,
            dtype=str(dtype),
            logical_type=logical_type(dtype),
            nullable=bool(frame[name].null_count()),
        )
        for name, dtype in frame.schema.items()
    ]


class DatasetRegistry:
    """Файловое хранилище снимков.

    Реализация намеренно простая: каталог на снимок, Parquet рядом с карточкой.
    Заменить её на базу можно, не трогая остальные слои, — наружу торчат только методы ниже.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or get_settings().datasets_directory

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------ импорт

    def import_package(self, package_path: Path) -> DatasetSnapshot:
        #пакет уже проверен по контракту: отпечаток сошёлся, схема согласована с данными
        settings = get_settings()
        schema_directory = settings.package_schema_directory
        schema_available = schema_directory.exists()
        loaded = read_package(package_path, schema_directory if schema_available else None)
        manifest = loaded.manifest
        warnings = list(loaded.warnings)

        if not schema_available:
            #пропуск проверки по JSON Schema — понижение строгости, и оно обязано быть
            #видимым: иначе пакет с некорректной schema.json пройдёт как полностью проверенный
            warnings.append(
                "JSON Schema контракта не найдена, метаданные пакета не проверены по схеме. "
                f"Ожидался каталог {schema_directory}."
            )

        return self._store(
            frame=loaded.frame,
            name=loaded.name,
            source="package",
            fingerprint=loaded.fingerprint,
            row_key=loaded.row_key,
            warnings=warnings,
            package_ref={
                "format": manifest.get("format"),
                "format_version": manifest.get("format_version"),
                "package_id": manifest.get("package_id"),
                "producer": manifest.get("producer"),
                "version": manifest.get("version"),
                "links": manifest.get("links"),
                "hints": manifest.get("hints"),
            },
        )

    def import_file(self, file_path: Path, name: str | None = None) -> DatasetSnapshot:
        #standalone-путь (D-13): читаем, но не редактируем. Отпечаток считается тем же
        #алгоритмом, что и для пакета, поэтому csv и parquet с одинаковым содержимым
        #дают один снимок
        frame = read_table(file_path)

        return self._store(
            frame=frame,
            name=name or file_path.stem,
            source="file",
            fingerprint=content_fingerprint(frame),
            row_key=[],
            warnings=[
                "Датасет загружен файлом, минуя DataArena: происхождение и преобразования неизвестны."
            ],
            package_ref=None,
        )

    # ------------------------------------------------------------------ чтение

    def list_snapshots(self) -> list[DatasetSnapshot]:
        if not self._root.exists():
            return []

        snapshots: list[DatasetSnapshot] = []

        for directory in self._root.iterdir():
            if directory.name.startswith(".staging_"):
                continue

            meta_path = directory / META_FILE

            if not directory.is_dir() or not meta_path.exists():
                continue

            try:
                snapshots.append(
                    DatasetSnapshot.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                #повреждённая карточка не должна ломать весь список, но и исчезать молча
                #не имеет права: снимок остаётся на диске, не попадает в поиск по отпечатку
                #и превращается в невидимый дубль при следующем импорте
                logger.warning("Карточка снимка %s не читается: %s", directory.name, error)
                continue

        snapshots.sort(key=lambda snapshot: snapshot.created_at, reverse=True)
        return snapshots

    def get(self, dataset_id: str) -> DatasetSnapshot:
        meta_path = self._directory(dataset_id) / META_FILE

        if not meta_path.exists():
            raise NotFoundError(f"Датасет {dataset_id} не найден.")

        return DatasetSnapshot.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))

    def data_path(self, dataset_id: str) -> Path:
        """Путь к файлу снимка.

        Нужен процессам обучения: они читают снимок сами, а не получают его через pickle.
        Кадр на сотни мегабайт, переданный аргументом процесса, копировался бы в память
        дважды на каждого контендера.
        """
        path = self._directory(dataset_id) / SNAPSHOT_FILE

        if not path.exists():
            raise NotFoundError(f"Файл данных снимка {dataset_id} отсутствует.")

        return path

    def load_frame(self, dataset_id: str, with_row_id: bool = False) -> pl.DataFrame:
        """Прочитать данные снимка.

        `__row_id__` добавляется только по явному запросу и только для тех мест, где нужны
        ссылки на конкретные строки. В матрицу признаков он не попадает никогда.
        """
        path = self._directory(dataset_id) / SNAPSHOT_FILE

        if not path.exists():
            raise NotFoundError(f"Файл данных снимка {dataset_id} отсутствует.")

        frame = pl.read_parquet(path)

        if with_row_id:
            return frame.with_row_index(name=ROW_ID)

        return frame

    def find_by_fingerprint(self, fingerprint: str) -> DatasetSnapshot | None:
        return next(
            (snapshot for snapshot in self.list_snapshots() if snapshot.fingerprint == fingerprint),
            None,
        )

    def delete(self, dataset_id: str) -> None:
        shutil.rmtree(self._directory(dataset_id))

    # ------------------------------------------------------------------ внутреннее

    def _store(
        self,
        frame: pl.DataFrame,
        *,
        name: str,
        source: Literal["package", "file"],
        fingerprint: str,
        row_key: list[str],
        warnings: list[str],
        package_ref: dict[str, Any] | None,
    ) -> DatasetSnapshot:
        _validate_frame(frame)
        existing = self.find_by_fingerprint(fingerprint)

        if existing is not None:
            #те же данные — тот же снимок. Повторный импорт не создаёт дубль и не переписывает
            #карточку: на неё уже могут ссылаться эксперименты
            return existing

        if ROW_ID in frame.columns:
            raise ValidationError(
                f"Колонка {ROW_ID} зарезервирована ModelArena и не может быть в датасете."
            )

        dataset_id = _new_dataset_id()
        directory = self._root / dataset_id
        #снимок собирается рядом и переносится одним движением: если процесс умрёт между
        #записью parquet и записью карточки, на месте снимка останется каталог без meta.json —
        #невидимый для списка, но занимающий место и ломающий поиск по отпечатку
        staging = self._root / f".staging_{dataset_id}"
        staging.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(staging / SNAPSHOT_FILE, compression="zstd")

        snapshot = DatasetSnapshot(
            dataset_id=dataset_id,
            name=name,
            source=source,
            fingerprint=fingerprint,
            fingerprint_algorithm="dataarena-logical-sha256-v1",
            row_count=frame.height,
            column_count=frame.width,
            columns=_describe_columns(frame),
            created_at=_utc_now(),
            row_key=row_key,
            package_ref=package_ref,
            warnings=warnings,
        )
        (staging / META_FILE).write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        try:
            os.replace(staging, directory)
        except OSError:
            #каталог с таким идентификатором уже существует — значит два импорта пришли
            #одновременно. Побеждает тот, кто успел; свой staging убираем за собой
            shutil.rmtree(staging, ignore_errors=True)
            existing = self.find_by_fingerprint(fingerprint)

            if existing is None:
                raise

            return existing

        return snapshot

    def _directory(self, dataset_id: str) -> Path:
        #идентификатор попадает в путь, поэтому проверяется по белому списку символов.
        #str.isalnum() здесь не годится: он пропускает Unicode-цифры и буквы, а те
        #на регистронезависимой файловой системе способны схлопнуться в один каталог
        if not DATASET_ID_PATTERN.match(dataset_id):
            raise ValidationError(f"Некорректный идентификатор датасета: {dataset_id}")

        return self._root / dataset_id


def _validate_frame(frame: pl.DataFrame) -> None:
    if frame.height < MIN_ROWS:
        raise ValidationError("В датасете меньше двух строк — обучать и оценивать нечего.")

    if frame.width < MIN_COLUMNS:
        raise ValidationError("Нужна минимум одна колонка признаков помимо целевой.")

    duplicates = [name for name in frame.columns if frame.columns.count(name) > 1]

    if duplicates:
        raise ValidationError(f"Дублирующиеся имена колонок: {sorted(set(duplicates))}")
