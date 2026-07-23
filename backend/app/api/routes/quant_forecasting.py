"""Read-only observability API for Phase 2 shadow forecasts."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.app.quant_forecasting.models import CalibrationReport, ForecastOutcome, QuantForecastResult
from backend.app.quant_forecasting.repository import QuantForecastRepository
from backend.app.quant_forecasting.service import QuantForecastService

router = APIRouter(prefix="/api/v1/quant-forecasts", tags=["quantitative-forecasting"])


def get_service(request: Request) -> QuantForecastService:
    return cast(QuantForecastService, request.app.state.quant_forecast_service)


def get_repository(request: Request) -> QuantForecastRepository:
    return cast(QuantForecastRepository, request.app.state.quant_forecast_repository)


Service = Annotated[QuantForecastService, Depends(get_service)]
Repository = Annotated[QuantForecastRepository, Depends(get_repository)]


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/latest", response_model=QuantForecastResult)
async def latest(repository: Repository, instrument: str = "XAUUSD") -> QuantForecastResult:
    value = await repository.latest_result(instrument)
    if value is None:
        raise HTTPException(404, "No shadow quantitative forecast is available")
    return value


@router.get("/calibration/latest", response_model=CalibrationReport)
async def latest_calibration(repository: Repository, model_name: str = "deterministic_xauusd_baseline") -> CalibrationReport:
    value = await repository.latest_calibration(model_name)
    if value is None:
        raise HTTPException(404, "No calibration report is available")
    return value


@router.get("/{result_id}/outcomes", response_model=list[ForecastOutcome])
async def outcomes(result_id: UUID, repository: Repository) -> list[ForecastOutcome]:
    return list(await repository.outcomes_for_result(result_id))
