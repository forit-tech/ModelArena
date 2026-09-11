"""Единый контракт ошибок ModelArena.

Формат ответа один на всё приложение:
`{"error": {"code", "message", "details"}, "detail": "..."}`.

Поле `detail` дублирует сообщение: так ответ остаётся читаемым и для клиентов,
которые разбирают только его.
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    #базовая ошибка приложения: несёт HTTP-статус и машинный код, а не только текст
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AppError):
    status_code = 400
    code = "validation_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class UnsupportedFormatError(AppError):
    status_code = 415
    code = "unsupported_format"


class DatasetReadError(AppError):
    status_code = 400
    code = "dataset_read_error"


class PackageError(AppError):
    """Пакет отвергнут проверкой контракта.

    Код приходит из матрицы §H спецификации (`package_integrity_failed`,
    `package_fingerprint_mismatch`, …) и передаётся наружу как есть: обе стороны
    договорились называть одну и ту же причину одинаково, и подменять код
    на собственный означало бы разорвать эту договорённость.
    """

    status_code = 400

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)
        self.code = code


def build_error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    #эта функция собирает единый ответ об ошибке для всех слоёв приложения
    return {
        "error": {"code": code, "message": message, "details": details or {}},
        "detail": message,
    }
