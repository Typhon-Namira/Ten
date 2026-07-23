"""Read-only Phase 3/4 AI reasoning and managed-signal observability."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from backend.app.ai_reasoning.repository import AIReasoningRepository
from backend.app.ai_reasoning.service import AIReasoningService
from backend.app.final_decision.repository import FinalDecisionRepository
from backend.app.final_decision.service import FinalDecisionService

router = APIRouter(prefix="/api/v1/ai-reasoning", tags=["ai-reasoning"])


def get_service(request: Request) -> AIReasoningService:
    return request.app.state.ai_reasoning_service


def get_repository(request: Request) -> AIReasoningRepository:
    return request.app.state.ai_reasoning_repository


Service = Annotated[AIReasoningService, Depends(get_service)]
Repository = Annotated[AIReasoningRepository, Depends(get_repository)]


def get_final_service(request: Request) -> FinalDecisionService:
    return request.app.state.final_decision_service


def get_final_repository(request: Request) -> FinalDecisionRepository:
    return request.app.state.final_decision_repository


FinalService = Annotated[FinalDecisionService, Depends(get_final_service)]
FinalRepository = Annotated[FinalDecisionRepository, Depends(get_final_repository)]


@router.get("/health")
async def health(service: Service, final_service: FinalService) -> dict[str, object]:
    return {**service.health(), "guardrails": final_service.health()}


@router.get("/latest")
async def latest(
    repository: Repository,
    service: Service,
    final_repository: FinalRepository,
    final_service: FinalService,
    instrument: str = "XAUUSD",
) -> dict[str, Any]:
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
    final_actions = {
        str(signal.signal_id): [
            item.model_dump(mode="json")
            for item in await final_repository.action_history(signal.signal_id)
        ]
        for signal in signals
    }
    publications = {
        str(signal.signal_id): (
            value.model_dump(mode="json")
            if (value := await final_repository.publication_for_signal(signal.signal_id))
            else None
        )
        for signal in signals
    }
    usage = await final_repository.usage_for_date(service.clock().date().isoformat())
    performance = await final_repository.latest_performance_report()
    readiness = await final_repository.latest_readiness_report()
    return {
        "forecast": forecast.model_dump(mode="json") if forecast else None,
        "proposal": proposal.model_dump(mode="json") if proposal else None,
        "managed_signals": [signal.model_dump(mode="json") for signal in signals],
        "signal_histories": histories,
        "final_actions": final_actions,
        "publications": publications,
        "llm_usage": {
            "request_count": sum(item.request_count for item in usage),
            "total_tokens": (
                sum(item.total_tokens or 0 for item in usage)
                if any(item.total_tokens is not None for item in usage)
                else None
            ),
            "successful_requests": sum(item.success for item in usage),
            "failed_requests": sum(not item.success for item in usage),
        },
        "performance": performance.model_dump(mode="json") if performance else None,
        "production_readiness": readiness.model_dump(mode="json") if readiness else None,
        "health": {**service.health(), "guardrails": final_service.health()},
    }
