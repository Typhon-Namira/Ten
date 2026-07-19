from fastapi import Request
from typing import cast

from backend.app.services import EngineRegistry, SignalRepository
from backend.app.engines.market_data_engine import MarketDataService
from backend.app.engines.smc_engine import SMCService
from backend.app.engines.liquidity_engine import LiquidityService
from backend.app.engines.volume_profile_engine import VolumeProfileService
from backend.app.engines.institutional_flow_engine import InstitutionalFlowService
from backend.app.engines.market_regime_engine import MarketRegimeService
from backend.app.engines.economic_calendar_engine import EconomicCalendarService
from backend.app.engines.ai_scoring_engine import AIScoringService
from backend.app.engines.signal_decision_engine import SignalDecisionService
from backend.app.engines.replay_engine import ReplayService
from backend.app.integration import FullSystemIntegrationService


def get_signal_repository(request: Request) -> SignalRepository:
    return cast(SignalRepository, request.app.state.signal_repository)


def get_engine_registry(request: Request) -> EngineRegistry:
    return cast(EngineRegistry, request.app.state.engine_registry)


def get_market_data_service(request: Request) -> MarketDataService:
    return cast(MarketDataService, request.app.state.market_data_service)


def get_smc_service(request: Request) -> SMCService:
    return cast(SMCService, request.app.state.smc_service)


def get_liquidity_service(request: Request) -> LiquidityService:
    return cast(LiquidityService, request.app.state.liquidity_service)


def get_volume_profile_service(request: Request) -> VolumeProfileService:
    return cast(VolumeProfileService, request.app.state.volume_profile_service)


def get_institutional_flow_service(request: Request) -> InstitutionalFlowService:
    return cast(InstitutionalFlowService, request.app.state.institutional_flow_service)


def get_market_regime_service(request: Request) -> MarketRegimeService:
    return cast(MarketRegimeService, request.app.state.market_regime_service)


def get_economic_calendar_service(request: Request) -> EconomicCalendarService:
    return cast(EconomicCalendarService, request.app.state.economic_calendar_service)


def get_ai_scoring_service(request: Request) -> AIScoringService:
    return cast(AIScoringService, request.app.state.ai_scoring_service)


def get_signal_decision_service(request: Request) -> SignalDecisionService:
    return cast(SignalDecisionService, request.app.state.signal_decision_service)


def get_replay_service(request: Request) -> ReplayService:
    return cast(ReplayService, request.app.state.replay_service)


def get_integration_service(request: Request) -> FullSystemIntegrationService:
    return cast(FullSystemIntegrationService, request.app.state.integration_service)
