from fastapi import Request

from backend.app.services import EngineRegistry, SignalRepository


def get_signal_repository(request: Request) -> SignalRepository:
    return request.app.state.signal_repository


def get_engine_registry(request: Request) -> EngineRegistry:
    return request.app.state.engine_registry

