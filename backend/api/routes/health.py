"""Проверка живости и версии контракта, который понимает этот backend."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.datasets.package import FINGERPRINT_ALGORITHM, SUPPORTED_FORMAT, SUPPORTED_MAJOR

router = APIRouter(tags=["health"])


@router.get("/health", summary="Живость сервиса и понимаемая версия контракта")
def health() -> dict[str, Any]:
    #версия контракта отдаётся наружу намеренно: при расхождении с DataArena причина
    #видна сразу, а не выясняется по коду отказа при первом импорте
    return {
        "status": "ok",
        "service": "modelarena",
        "version": "0.1.0",
        "dataset_package": {
            "format": SUPPORTED_FORMAT,
            "major_version": SUPPORTED_MAJOR,
            "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        },
    }
