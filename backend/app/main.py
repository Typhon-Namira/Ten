"""TEN FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import api_router
from backend.app.core.config import get_settings
from backend.app.core.exceptions import TenError
from backend.app.core.logging import configure_logging
from backend.app.core.config import YamlConfigRepository
from backend.app.engines.market_data_engine import build_market_data_service
from backend.app.engines.market_data_engine.config import MarketDataConfig
from backend.app.engines.smc_engine import SMCConfig, SMCService
from backend.app.engines.smc_engine.repository import InMemorySMCRepository, SMCRepository, SqlAlchemySMCRepository
from backend.app.engines.liquidity_engine import InMemoryLiquidityRepository, LiquidityConfig, LiquidityRepository, LiquidityService, SqlAlchemyLiquidityRepository
from backend.app.core.database.base import Base
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from backend.app.services import InMemorySignalRepository, PipelineManager, build_engine_registry


def create_app() -> FastAPI:
    """Construct the HTTP application and inject its adapters."""

    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configs = YamlConfigRepository()
        app.state.signal_repository = InMemorySignalRepository()
        app.state.engine_registry = build_engine_registry(configs=configs)
        app.state.pipeline_manager = PipelineManager.from_yaml(app.state.engine_registry, configs)
        market_config = configs.load_model("market_data", MarketDataConfig)
        app.state.market_data_service = build_market_data_service(market_config)
        smc_config = configs.load_model("smc", SMCConfig)
        app.state.smc_database_engine = None
        app.state.smc_database_session = None
        app.state.liquidity_database_session = None
        repository: SMCRepository = InMemorySMCRepository()
        database_engine = None
        try:
            database_engine = create_async_engine(settings.database_url, pool_pre_ping=True, connect_args={"timeout": 3})
            async with database_engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(database_engine, expire_on_commit=False)
            database_session = session_factory()
            repository = SqlAlchemySMCRepository(database_session)
            app.state.smc_database_engine = database_engine
            app.state.smc_database_session = database_session
            app.state.liquidity_database_session = session_factory()
            logger.info("SMC durable persistence activated", extra={"engine": "smc", "adapter": "sqlalchemy"})
        except Exception as exc:
            if database_engine is not None:
                await database_engine.dispose()
            logger.warning("SMC database unavailable; using bounded in-memory persistence", extra={"engine": "smc", "error_type": type(exc).__name__})
        app.state.smc_service = SMCService(
            app.state.market_data_service,
            app.state.pipeline_manager.event_bus,
            app.state.pipeline_manager.feature_store,
            smc_config,
            repository,
        )
        await app.state.smc_service.restore()
        liquidity_config = configs.load_model("liquidity", LiquidityConfig)
        liquidity_repository: LiquidityRepository = InMemoryLiquidityRepository()
        liquidity_mode = "memory"
        if app.state.liquidity_database_session is not None:
            liquidity_repository = SqlAlchemyLiquidityRepository(app.state.liquidity_database_session)
            liquidity_mode = "sqlalchemy"
        app.state.liquidity_service = LiquidityService(app.state.market_data_service, app.state.smc_service, app.state.pipeline_manager.event_bus, app.state.pipeline_manager.feature_store, liquidity_config, liquidity_repository, liquidity_mode)
        await app.state.liquidity_service.restore()
        try:
            yield
        finally:
            await app.state.market_data_service.close()
            if app.state.liquidity_database_session is not None:
                await app.state.liquidity_database_session.close()
            if app.state.smc_database_session is not None:
                await app.state.smc_database_session.close()
            if app.state.smc_database_engine is not None:
                await app.state.smc_database_engine.dispose()

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
