"""Сборка HTTP-приложения.

Фабрика, а не модульный синглтон: каждому тесту нужен свой экземпляр со своим каталогом
артефактов, иначе тесты начинают видеть данные друг друга.

Ошибки приводятся к одному формату на все слои. Без этого FastAPI, Starlette и предметный
код отвечали бы тремя разными структурами, и фронт разбирал бы каждую отдельно.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.routes import arena, datasets, health, tasks
from backend.core.config import get_settings
from backend.core.errors import AppError, build_error_payload

logger = logging.getLogger("modelarena")

API_DESCRIPTION = """
Лаборатория честного сравнения ML-моделей на подготовленном датасете.

**Принцип.** Каждое решение системы объясняется числами и может быть переопределено.
Там, где вывод сделать нельзя, ModelArena пишет «недостаточно данных», а не показывает
красивое неверное число.

**Формат ошибок.** `{"error": {"code", "message", "details"}, "detail": "..."}`.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ModelArena API", version="0.1.0", description=API_DESCRIPTION.strip())

    #без этого браузер заблокирует запросы с Vite-сервера: фронт и backend на разных портах
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    for module in (health, datasets, tasks, arena):
        app.include_router(module.router, prefix="/api")

    _register_error_handlers(app)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=error.status_code,
            content=build_error_payload(error.code, error.message, error.details),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(
        request: Request,  # noqa: ARG001
        error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=build_error_payload(
                "request_validation_error",
                "Запрос не соответствует контракту API.",
                {"errors": _safe_validation_errors(error)},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(
        request: Request,  # noqa: ARG001
        error: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=build_error_payload("http_error", str(error.detail)),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, error: Exception) -> JSONResponse:
        #стек уходит в лог целиком, наружу — короткое сообщение без внутренностей
        logger.exception("Необработанная ошибка на %s", request.url.path, exc_info=error)
        return JSONResponse(
            status_code=500,
            content=build_error_payload(
                "internal_error",
                "Внутренняя ошибка сервера. Подробности записаны в лог backend.",
            ),
        )


def _safe_validation_errors(error: RequestValidationError) -> list[dict[str, str]]:
    #из ошибок валидации наружу идут только место и текст: входные значения могут содержать
    #пользовательские данные, которым не место в ответе
    return [
        {
            "location": ".".join(str(part) for part in item.get("loc", [])),
            "message": str(item.get("msg", "")),
        }
        for item in error.errors()
    ]


app = create_app()
