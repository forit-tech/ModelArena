"""Настройки приложения.

Каждое поле имеет рабочее значение по умолчанию: ModelArena обязана полностью работать
без единой переменной окружения. Переменные читаются в одном месте, чтобы модули
не разбирали `os.environ` вразнобой и не расходились в дефолтах.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_MAX_UPLOAD_MEGABYTES = 500
#предел на данные — операционный, а не контрактный: контракт ограничивает только метаданные,
#потому что датасет законно бывает огромным, а Parquet читается частями
DEFAULT_MAX_PACKAGE_DATA_MEGABYTES = 5 * 1024


@dataclass(frozen=True)
class Settings:
    artifacts_directory: Path
    contracts_directory: Path
    max_upload_bytes: int
    max_package_data_bytes: int
    allowed_origins: tuple[str, ...]

    @property
    def datasets_directory(self) -> Path:
        return self.artifacts_directory / "datasets"

    @property
    def runs_directory(self) -> Path:
        return self.artifacts_directory / "runs"

    @property
    def package_schema_directory(self) -> Path:
        #JSON Schema контракта: зеркало, побайтово идентичное каталогу владельца
        return self.contracts_directory / "dataset-package" / "v1" / "schema"


def _read_int(variable_name: str, default_value: int) -> int:
    #опечатка в .env не должна мешать приложению стартовать: молча берём значение по умолчанию
    raw_value = os.environ.get(variable_name)

    if raw_value is None:
        return default_value

    try:
        parsed = int(raw_value)
    except ValueError:
        return default_value

    return parsed if parsed > 0 else default_value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    raw_origins = os.environ.get("MARENA_ALLOWED_ORIGINS")

    return Settings(
        artifacts_directory=Path(
            os.environ.get("MARENA_ARTIFACTS_DIR") or (project_root / "artifacts")
        ).resolve(),
        contracts_directory=Path(
            os.environ.get("MARENA_CONTRACTS_DIR") or (project_root / "contracts")
        ).resolve(),
        max_upload_bytes=_read_int("MARENA_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MEGABYTES) * 1024 * 1024,
        max_package_data_bytes=_read_int(
            "MARENA_MAX_PACKAGE_DATA_MB", DEFAULT_MAX_PACKAGE_DATA_MEGABYTES
        )
        * 1024
        * 1024,
        allowed_origins=(
            tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
            if raw_origins
            else ("http://localhost:5173", "http://127.0.0.1:5173")
        ),
    )
