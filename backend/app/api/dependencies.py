from fastapi import Request
from typing import cast

from backend.app.services import EngineRegistry, SignalRepository
from backend.app.engines.market_data_engine import MarketDataService


def get_signal_repository(request: Request) -> SignalRepository:
    return cast(SignalRepository, request.app.state.signal_repository)


def get_engine_registry(request: Request) -> EngineRegistry:
    return cast(EngineRegistry, request.app.state.engine_registry)


def get_market_data_service(request: Request) -> MarketDataService:
    return cast(MarketDataService, request.app.state.market_data_service)
