"""TEN FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
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
from backend.app.engines.volume_profile_engine import InMemoryVolumeProfileRepository, SqlAlchemyVolumeProfileRepository, VolumeProfileConfig, VolumeProfileRepository, VolumeProfileService
from backend.app.engines.institutional_flow_engine import InMemoryInstitutionalFlowRepository, InstitutionalFlowConfig, InstitutionalFlowRepository, InstitutionalFlowService, SqlAlchemyInstitutionalFlowRepository
from backend.app.engines.market_regime_engine import InMemoryMarketRegimeRepository, MarketRegimeConfig, MarketRegimeRepository, MarketRegimeService, SqlAlchemyMarketRegimeRepository
from backend.app.engines.economic_calendar_engine import EconomicCalendarConfig, EconomicCalendarRepository, EconomicCalendarService, InMemoryEconomicCalendarRepository, SqlAlchemyEconomicCalendarRepository, build_providers
from backend.app.engines.ai_scoring_engine import AIScoringConfig, AIScoringRepository, AIScoringService, InMemoryAIScoringRepository, SqlAlchemyAIScoringRepository
from backend.app.engines.signal_decision_engine import InMemorySignalDecisionRepository, SignalDecisionConfig, SignalDecisionRepository, SignalDecisionService, SqlAlchemySignalDecisionRepository
from backend.app.engines.replay_engine import (
    HistoricalSourceRegistry,
    InMemoryHistoricalSource,
    InMemoryReplayRepository,
    ReplayConfig,
    ReplayCoordinator,
    ReplayDatasetReference,
    ReplayDatasetRegistry,
    ReplayRepository,
    ReplayService,
    ReplayWorker,
    SqlAlchemyEconomicRevisionSource,
    SqlAlchemyHistoricalCandleSource,
    SqlAlchemyReplayRepository,
    dataset_manifest_hash,
    production_replay_registry,
)
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
        app.state.volume_profile_database_session = None
        app.state.institutional_flow_database_session = None
        app.state.market_regime_database_session = None
        app.state.economic_calendar_database_session = None
        app.state.ai_scoring_database_session = None
        app.state.signal_decision_database_session = None
        app.state.replay_database_session = None
        app.state.replay_source_database_session = None
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
            app.state.volume_profile_database_session = session_factory()
            app.state.institutional_flow_database_session = session_factory()
            app.state.market_regime_database_session = session_factory()
            app.state.economic_calendar_database_session = session_factory()
            app.state.ai_scoring_database_session = session_factory()
            app.state.signal_decision_database_session = session_factory()
            app.state.replay_database_session = session_factory()
            app.state.replay_source_database_session = session_factory()
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
        volume_profile_config = configs.load_model("volume_profile", VolumeProfileConfig)
        volume_profile_repository: VolumeProfileRepository = InMemoryVolumeProfileRepository()
        volume_profile_mode = "memory"
        if app.state.volume_profile_database_session is not None:
            volume_profile_repository = SqlAlchemyVolumeProfileRepository(app.state.volume_profile_database_session)
            volume_profile_mode = "sqlalchemy"
        app.state.volume_profile_service = VolumeProfileService(app.state.market_data_service, app.state.smc_service, app.state.liquidity_service, app.state.pipeline_manager.event_bus, app.state.pipeline_manager.feature_store, volume_profile_config, volume_profile_repository, volume_profile_mode)
        await app.state.volume_profile_service.restore()
        flow_config = configs.load_model("flow", InstitutionalFlowConfig)
        flow_repository: InstitutionalFlowRepository = InMemoryInstitutionalFlowRepository()
        flow_mode = "memory"
        if app.state.institutional_flow_database_session is not None:
            flow_repository = SqlAlchemyInstitutionalFlowRepository(app.state.institutional_flow_database_session)
            flow_mode = "sqlalchemy"
        app.state.institutional_flow_service = InstitutionalFlowService(app.state.market_data_service, app.state.smc_service, app.state.liquidity_service, app.state.volume_profile_service, app.state.pipeline_manager.event_bus, app.state.pipeline_manager.feature_store, flow_config, flow_repository, flow_mode)
        await app.state.institutional_flow_service.restore()
        regime_config = configs.load_model("market_regime", MarketRegimeConfig)
        regime_repository: MarketRegimeRepository = InMemoryMarketRegimeRepository()
        regime_mode = "memory"
        if app.state.market_regime_database_session is not None:
            regime_repository = SqlAlchemyMarketRegimeRepository(app.state.market_regime_database_session)
            regime_mode = "sqlalchemy"
        app.state.market_regime_service = MarketRegimeService(app.state.market_data_service, app.state.smc_service, app.state.liquidity_service, app.state.volume_profile_service, app.state.institutional_flow_service, app.state.pipeline_manager.event_bus, app.state.pipeline_manager.feature_store, regime_config, regime_repository, regime_mode)
        await app.state.market_regime_service.restore()
        calendar_config = configs.load_model("economic_calendar", EconomicCalendarConfig)
        calendar_repository: EconomicCalendarRepository = InMemoryEconomicCalendarRepository()
        calendar_mode = "memory"
        if app.state.economic_calendar_database_session is not None:
            calendar_repository = SqlAlchemyEconomicCalendarRepository(app.state.economic_calendar_database_session)
            calendar_mode = "postgresql"
        calendar_providers = build_providers(calendar_config.providers)
        app.state.economic_calendar_service = EconomicCalendarService(app.state.pipeline_manager.event_bus, app.state.pipeline_manager.feature_store, calendar_config, calendar_repository, calendar_providers, calendar_mode)
        await app.state.economic_calendar_service.start()
        ai_config = configs.load_model("ai_scoring", AIScoringConfig)
        ai_repository: AIScoringRepository = InMemoryAIScoringRepository()
        ai_mode = "memory"
        if app.state.ai_scoring_database_session is not None:
            ai_repository = SqlAlchemyAIScoringRepository(app.state.ai_scoring_database_session)
            ai_mode = "postgresql"
        app.state.ai_scoring_service = AIScoringService(
            ai_repository,
            app.state.pipeline_manager.event_bus,
            app.state.pipeline_manager.feature_store,
            ai_config,
            market_data=app.state.market_data_service,
            smc=app.state.smc_service,
            liquidity=app.state.liquidity_service,
            volume_profile=app.state.volume_profile_service,
            institutional_flow=app.state.institutional_flow_service,
            market_regime=app.state.market_regime_service,
            economic_calendar=app.state.economic_calendar_service,
            repository_mode=ai_mode,
        )
        await app.state.ai_scoring_service.start()
        decision_config = configs.load_model("signal_decision", SignalDecisionConfig)
        decision_repository: SignalDecisionRepository = InMemorySignalDecisionRepository()
        decision_mode = "memory"
        if app.state.signal_decision_database_session is not None:
            decision_repository = SqlAlchemySignalDecisionRepository(app.state.signal_decision_database_session)
            decision_mode = "postgresql"
        app.state.signal_decision_service = SignalDecisionService(
            decision_repository,
            app.state.ai_scoring_service,
            app.state.pipeline_manager.event_bus,
            app.state.pipeline_manager.feature_store,
            decision_config,
            economic_calendar=app.state.economic_calendar_service,
            market_regime=app.state.market_regime_service,
            repository_mode=decision_mode,
        )
        await app.state.signal_decision_service.start()
        replay_config = configs.load_model("replay", ReplayConfig)
        replay_repository: ReplayRepository = InMemoryReplayRepository()
        replay_mode = "memory"
        replay_sources = HistoricalSourceRegistry((InMemoryHistoricalSource("historical_candles", ()),))
        if app.state.replay_database_session is not None and app.state.replay_source_database_session is not None:
            replay_repository = SqlAlchemyReplayRepository(app.state.replay_database_session)
            replay_sources = HistoricalSourceRegistry(
                (
                    SqlAlchemyHistoricalCandleSource(app.state.replay_source_database_session),
                    SqlAlchemyEconomicRevisionSource(app.state.replay_source_database_session),
                )
            )
            replay_mode = "postgresql"
        dataset_created_at = datetime(2026, 7, 19, tzinfo=UTC)
        dataset_id = "ten-historical-postgres"
        dataset_version = "2026-07-19"
        dataset = ReplayDatasetReference(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            source_name="historical_candles",
            created_at=dataset_created_at,
            available_from=datetime(2000, 1, 1, tzinfo=UTC),
            available_until=dataset_created_at,
            manifest_hash=dataset_manifest_hash(dataset_id, dataset_version, dataset_created_at, "historical_candles"),
        )
        replay_coordinator = ReplayCoordinator(
            replay_repository,
            ReplayDatasetRegistry((dataset,)),
            replay_sources,
            production_replay_registry(),
            replay_config,
        )
        app.state.replay_service = ReplayService(replay_repository, replay_coordinator, app.state.pipeline_manager.event_bus, replay_config, repository_mode=replay_mode)
        app.state.replay_worker = ReplayWorker(app.state.replay_service, replay_config)
        await app.state.replay_service.start()
        if replay_config.worker.enabled and replay_config.worker.embedded_api_worker:
            app.state.replay_worker.start()
        try:
            yield
        finally:
            await app.state.replay_worker.stop()
            await app.state.replay_service.stop()
            await app.state.signal_decision_service.stop()
            await app.state.ai_scoring_service.stop()
            await app.state.economic_calendar_service.stop()
            await app.state.market_data_service.close()
            if app.state.economic_calendar_database_session is not None:
                await app.state.economic_calendar_database_session.close()
            if app.state.ai_scoring_database_session is not None:
                await app.state.ai_scoring_database_session.close()
            if app.state.signal_decision_database_session is not None:
                await app.state.signal_decision_database_session.close()
            if app.state.replay_source_database_session is not None:
                await app.state.replay_source_database_session.close()
            if app.state.replay_database_session is not None:
                await app.state.replay_database_session.close()
            if app.state.market_regime_database_session is not None:
                await app.state.market_regime_database_session.close()
            if app.state.institutional_flow_database_session is not None:
                await app.state.institutional_flow_database_session.close()
            if app.state.volume_profile_database_session is not None:
                await app.state.volume_profile_database_session.close()
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
