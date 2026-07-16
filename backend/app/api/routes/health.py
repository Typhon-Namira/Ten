from fastapi import APIRouter, Depends

from backend.app.api.schemas import HealthResponse
from backend.app.core.config import Settings, get_settings

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness endpoint with no external-service dependency."""

    return HealthResponse(status="healthy", service=settings.app_name, environment=settings.environment, version="0.1.0")

