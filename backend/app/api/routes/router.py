from fastapi import APIRouter

from .health import router as health_router
from .market_data import router as market_data_router
from .signals import router as signals_router
from .smc import router as smc_router
from .liquidity import router as liquidity_router
from .volume_profile import router as volume_profile_router
from .institutional_flow import router as institutional_flow_router
from .market_regime import router as market_regime_router
from .economic_calendar import router as economic_calendar_router
from .ai_scoring import router as ai_scoring_router
from .signal_decisions import router as signal_decisions_router
from .replays import router as replays_router
from .status import router as status_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(market_data_router)
api_router.include_router(signals_router)
api_router.include_router(smc_router)
api_router.include_router(liquidity_router)
api_router.include_router(volume_profile_router)
api_router.include_router(institutional_flow_router)
api_router.include_router(market_regime_router)
api_router.include_router(economic_calendar_router)
api_router.include_router(ai_scoring_router)
api_router.include_router(signal_decisions_router)
api_router.include_router(replays_router)
api_router.include_router(status_router)
