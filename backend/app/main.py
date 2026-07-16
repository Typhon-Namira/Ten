"""TEN FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import api_router
from backend.app.core.config import get_settings
from backend.app.core.exceptions import TenError
from backend.app.core.logging import configure_logging
from backend.app.core.config import YamlConfigRepository
from backend.app.services import InMemorySignalRepository, PipelineManager, build_engine_registry


def create_app() -> FastAPI:
    """Construct the HTTP application and inject its adapters."""

    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configs = YamlConfigRepository()
        app.state.signal_repository = InMemorySignalRepository()
        app.state.engine_registry = build_engine_registry(configs=configs)
        app.state.pipeline_manager = PipelineManager.from_yaml(app.state.engine_registry, configs)
        yield

    application = FastAPI(
        title="TEN Market Intelligence API",
        description="Analysis and scenario intelligence for XAU/USD. No order execution.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    application.include_router(api_router, prefix=settings.api_prefix)

    @application.exception_handler(TenError)
    async def handle_ten_error(_: Request, exc: TenError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)})

    return application


app = create_app()
