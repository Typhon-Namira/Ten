from fastapi import APIRouter

from .health import router as health_router
from .market_data import router as market_data_router
from .signals import router as signals_router
from .status import router as status_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(market_data_router)
api_router.include_router(signals_router)
api_router.include_router(status_router)
