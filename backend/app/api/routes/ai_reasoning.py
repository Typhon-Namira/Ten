"""Read-only Phase 3/4 AI reasoning and managed-signal observability."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from backend.app.ai_reasoning.repository import AIReasoningRepository
from backend.app.ai_reasoning.service import AIReasoningService

router = APIRouter(prefix="/api/v1/ai-reasoning", tags=["ai-reasoning"])


def get_service(request: Request) -> AIReasoningService:
    return request.app.state.ai_reasoning_service


def get_repository(request: Request) -> AIReasoningRepository:
    return request.app.state.ai_reasoning_repository


Service = Annotated[AIReasoningService, Depends(get_service)]
Repository = Annotated[AIReasoningRepository, Depends(get_repository)]


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/latest")
async def latest(repository: Repository, service: Service, instrument: str = "XAUUSD") -> dict[str, Any]:
    forecast = await repository.latest_forecast(instrument)
    proposal = await repository.latest_proposal()
    signals = await repository.active_signals(instrument)
    histories = {
        str(signal.signal_id): {
            key: [item.model_dump(mode="json") for item in values]
            for key, values in (await repository.signal_history(signal.signal_id)).items()
        }
        for signal in signals
    }
    return {
        "forecast": forecast.model_dump(mode="json") if forecast else None,
        "proposal": proposal.model_dump(mode="json") if proposal else None,
        "managed_signals": [signal.model_dump(mode="json") for signal in signals],
        "signal_histories": histories,
        "health": service.health(),
    }
