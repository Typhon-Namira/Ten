"""Read-only AI market-analysis and legacy lifecycle observability."""

from datetime import UTC, datetime
from typing import Annotated, Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.app.ai_reasoning.analysis import AIMarketAnalysis, AnalysisStatus
from backend.app.ai_reasoning.repository import AIReasoningRepository
from backend.app.ai_reasoning.service import AIReasoningService
from backend.app.final_decision.repository import FinalDecisionRepository
from backend.app.final_decision.service import FinalDecisionService

router = APIRouter(prefix="/api/v1/ai-reasoning", tags=["ai-reasoning"])


def _runtime_state(request: Request) -> dict[str, Any]:
    flags = request.app.state.engine_registry.context.feature_flags.snapshot()
    if flags.get("ai_signal_publication", False):
        profile = "analytical_live"
    elif flags.get("ai_centric_shadow_mode", False):
        profile = "shadow"
    else:
        profile = "safe_test"
    return {
        "operating_profile": profile,
        "feature_flags": flags,
        "analytical_only": True,
        "broker_execution_available": False,
    }


def get_service(request: Request) -> AIReasoningService:
    return cast(AIReasoningService, request.app.state.ai_reasoning_service)


def get_repository(request: Request) -> AIReasoningRepository:
    return cast(AIReasoningRepository, request.app.state.ai_reasoning_repository)


Service = Annotated[AIReasoningService, Depends(get_service)]
Repository = Annotated[AIReasoningRepository, Depends(get_repository)]


def get_final_service(request: Request) -> FinalDecisionService:
    return cast(FinalDecisionService, request.app.state.final_decision_service)


def get_final_repository(request: Request) -> FinalDecisionRepository:
    return cast(FinalDecisionRepository, request.app.state.final_decision_repository)


FinalService = Annotated[FinalDecisionService, Depends(get_final_service)]
FinalRepository = Annotated[FinalDecisionRepository, Depends(get_final_repository)]


@router.get("/health")
async def health(request: Request, service: Service, final_service: FinalService) -> dict[str, object]:
    return {**service.health(), "guardrails": final_service.health(), "runtime": _runtime_state(request)}


@router.get("/latest")
async def latest(
    request: Request,
    repository: Repository,
    service: Service,
    final_repository: FinalRepository,
    final_service: FinalService,
    instrument: str = "XAUUSD",
) -> dict[str, Any]:
    analysis = await repository.latest_analysis(instrument)
    forecast = await repository.latest_forecast(instrument)
    proposal = await repository.latest_proposal()
    signals = await repository.active_signals(instrument)
    histories = {
        str(signal.signal_id): {
            key: [cast(Any, item).model_dump(mode="json") for item in values]
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
    policy_usage = tuple(
        item
        for item in usage
        if item.generation_parameters.get("telemetry_policy") == "five_minute_v1"
    )
    legacy_usage = tuple(item for item in usage if item not in policy_usage)

    def usage_summary(rows: tuple[Any, ...]) -> dict[str, int | None]:
        return {
            "provider_http_calls": sum(item.request_count for item in rows),
            "total_tokens": (
                sum(item.total_tokens or 0 for item in rows)
                if any(item.total_tokens is not None for item in rows)
                else None
            ),
            "successful_requests": sum(item.success for item in rows),
            "failed_requests": sum(not item.success for item in rows),
        }

    def usage_parameter(name: str) -> int:
        return sum(
            int(item.generation_parameters.get(name, 0))
            for item in usage
            if isinstance(item.generation_parameters.get(name, 0), (int, float))
        )

    performance = await final_repository.latest_performance_report()
    readiness = await final_repository.latest_readiness_report()
    health = service.health()
    provider_states = health.get("providers")
    if isinstance(provider_states, dict):
        for account_id, provider_state in provider_states.items():
            if not isinstance(account_id, str) or not isinstance(provider_state, dict):
                continue
            provider_state["calls_today"] = usage_parameter(f"{account_id}_calls")
            provider_state["successful_analyses"] = sum(
                int(item.success)
                for item in usage
                if item.generation_parameters.get("provider") == account_id
            )
            provider_state["provider_failures"] = usage_parameter(
                f"{account_id}_provider_failures"
            )
            provider_state["rate_limit_failures"] = usage_parameter(
                f"{account_id}_rate_limit_failures"
            )
            provider_state["quota_failures"] = usage_parameter(
                f"{account_id}_quota_failures"
            )
            provider_state["token_usage"] = {
                "input_tokens": usage_parameter(f"{account_id}_input_tokens"),
                "output_tokens": usage_parameter(f"{account_id}_output_tokens"),
                "total_tokens": usage_parameter(f"{account_id}_total_tokens"),
            }
    return {
        "analysis": analysis.model_dump(mode="json") if analysis else None,
        "forecast": forecast.model_dump(mode="json") if forecast else None,
        "proposal": proposal.model_dump(mode="json") if proposal else None,
        "managed_signals": [signal.model_dump(mode="json") for signal in signals],
        "signal_histories": histories,
        "final_actions": final_actions,
        "publications": publications,
        "llm_usage": {
            "request_count": sum(item.request_count for item in usage),
            "provider_http_calls": sum(item.request_count for item in usage),
            "groq_calls": usage_parameter("groq_calls"),
            "retries": usage_parameter("retry_attempts"),
            "schema_corrections": usage_parameter("schema_corrections"),
            "provider_failures": usage_parameter("provider_failure"),
            "validation_failures": usage_parameter("validation_failure"),
            "total_tokens": (
                sum(item.total_tokens or 0 for item in usage)
                if any(item.total_tokens is not None for item in usage)
                else None
            ),
            "successful_requests": sum(item.success for item in usage),
            "failed_requests": sum(not item.success for item in usage),
            "legacy_cumulative_daily": usage_summary(legacy_usage),
            "five_minute_policy": usage_summary(policy_usage),
        },
        "performance": performance.model_dump(mode="json") if performance else None,
        "production_readiness": readiness.model_dump(mode="json") if readiness else None,
        "runtime": _runtime_state(request),
        "health": {**health, "guardrails": final_service.health()},
    }


@router.get("/analyses", response_model=list[AIMarketAnalysis])
async def analyses(
    repository: Repository,
    instrument: str = "XAUUSD",
    timeframe: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    status: AnalysisStatus | None = None,
    provider: str | None = None,
    cursor: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[AIMarketAnalysis]:
    for name, value in (("start", start), ("end", end)):
        if value is not None and value.tzinfo is None:
            raise HTTPException(422, f"{name} must include a timezone")
    if start is not None:
        start = start.astimezone(UTC)
    if end is not None:
        end = end.astimezone(UTC)
    if start is not None and end is not None and start > end:
        raise HTTPException(422, "start must precede end")
    return list(
        await repository.list_analyses(
            instrument.upper(),
            timeframe,
            start,
            end,
            status,
            provider,
            cursor,
            limit,
        )
    )


@router.get("/analyses/{analysis_id}", response_model=AIMarketAnalysis)
async def analysis_detail(
    analysis_id: UUID,
    repository: Repository,
) -> AIMarketAnalysis:
    value = await repository.get_analysis(analysis_id)
    if value is None:
        raise HTTPException(404, "AI market analysis not found")
    return value


@router.get("/analyses/{analysis_id}/temporal-context")
async def temporal_context(
    analysis_id: UUID,
    repository: Repository,
    service: Service,
) -> dict[str, Any]:
    value = await repository.get_analysis(analysis_id)
    if value is None:
        raise HTTPException(404, "AI market analysis not found")
    validated = await service._validated_analysis(value)
    if validated is None:
        raise HTTPException(409, "AI market analysis is not valid")
    return {
        "temporal_context": validated.temporal_context.model_dump(mode="json"),
        "temporal_metrics": validated.temporal_metrics.model_dump(mode="json"),
    }
