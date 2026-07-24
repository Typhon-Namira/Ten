"""Versioned, bounded persistence contract for AI reasoning request history.

The internal :class:`AIReasoningRequest` is intentionally large and is never the
database read model.  The provider-facing :class:`LLMAnalysisContext` is compact,
but it is also not reconstructed as the internal request.  This module provides
the stable history/dashboard DTO between those two boundaries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from .llm_context import LLMAnalysisContext
from .models import AIReasoningRequest

PERSISTED_REQUEST_PAYLOAD_TYPE = "ai_reasoning_request_snapshot"
PERSISTED_REQUEST_SCHEMA_VERSION = "1.0"


class _RequestRecord(Protocol):
    request_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    quantitative_forecast_id: UUID
    instrument: str
    analysis_timestamp: datetime
    prompt_version: str
    model_identifier: str
    payload: dict[str, Any]
    created_at: datetime


class PersistedAIReasoningRequest(BaseModel):
    """Small read model used by dashboard/history consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    quantitative_forecast_id: UUID
    instrument: str
    analysis_timestamp: datetime
    prompt_version: str
    model_identifier: str
    created_at: datetime
    compatibility_status: Literal["compatible", "incompatible"]
    payload_format: Literal[
        "versioned_compact",
        "legacy_compact_context",
        "legacy_full_request",
        "incompatible",
    ]
    payload_schema_version: str | None = None
    context_schema_version: str | None = None
    compatibility_reason: str | None = None

    @field_validator("analysis_timestamp", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("persisted request timestamps must be timezone-aware")
        return value.astimezone(UTC)


def persisted_request_payload(
    request: AIReasoningRequest,
    context: LLMAnalysisContext,
) -> dict[str, Any]:
    """Build the compact, explicitly discriminated JSONB payload."""

    return {
        "payload_type": PERSISTED_REQUEST_PAYLOAD_TYPE,
        "payload_schema_version": PERSISTED_REQUEST_SCHEMA_VERSION,
        "context_schema_version": context.schema_version,
        "request_id": str(request.request_id),
        "cycle_id": str(request.cycle_id),
        "market_state_id": str(request.market_state_id),
        "quantitative_forecast_id": str(request.quantitative_forecast_id),
        "instrument": request.instrument,
        "analysis_timestamp": request.analysis_timestamp.isoformat(),
        "knowledge_cutoff": request.knowledge_cutoff.isoformat(),
        "prompt_version": request.prompt_version,
        "reasoning_policy_version": request.reasoning_policy_version,
        "model_identifier": request.model_identifier,
        "context": context.model_dump(mode="json"),
    }


def persisted_request_from_domain(request: AIReasoningRequest) -> PersistedAIReasoningRequest:
    """Create the same compact read model for the in-memory repository."""

    return PersistedAIReasoningRequest(
        request_id=request.request_id,
        cycle_id=request.cycle_id,
        market_state_id=request.market_state_id,
        quantitative_forecast_id=request.quantitative_forecast_id,
        instrument=request.instrument,
        analysis_timestamp=request.analysis_timestamp,
        prompt_version=request.prompt_version,
        model_identifier=request.model_identifier,
        created_at=request.created_at,
        compatibility_status="compatible",
        payload_format="versioned_compact",
        payload_schema_version=PERSISTED_REQUEST_SCHEMA_VERSION,
        context_schema_version=LLMAnalysisContext.model_fields["schema_version"].default,
    )


def _snapshot(
    record: _RequestRecord,
    *,
    compatibility_status: Literal["compatible", "incompatible"],
    payload_format: Literal[
        "versioned_compact",
        "legacy_compact_context",
        "legacy_full_request",
        "incompatible",
    ],
    payload_schema_version: str | None = None,
    context_schema_version: str | None = None,
    compatibility_reason: str | None = None,
) -> PersistedAIReasoningRequest:
    return PersistedAIReasoningRequest(
        request_id=record.request_id,
        cycle_id=record.cycle_id,
        market_state_id=record.market_state_id,
        quantitative_forecast_id=record.quantitative_forecast_id,
        instrument=record.instrument,
        analysis_timestamp=record.analysis_timestamp,
        prompt_version=record.prompt_version,
        model_identifier=record.model_identifier,
        created_at=record.created_at,
        compatibility_status=compatibility_status,
        payload_format=payload_format,
        payload_schema_version=payload_schema_version,
        context_schema_version=context_schema_version,
        compatibility_reason=compatibility_reason,
    )


def decode_persisted_request(record: _RequestRecord) -> PersistedAIReasoningRequest:
    """Decode current and historical JSONB shapes without conflating contracts."""

    payload = record.payload
    payload_type = payload.get("payload_type")
    context_payload = payload.get("context")

    if payload_type == PERSISTED_REQUEST_PAYLOAD_TYPE:
        version = payload.get("payload_schema_version")
        if version != PERSISTED_REQUEST_SCHEMA_VERSION:
            return _snapshot(
                record,
                compatibility_status="incompatible",
                payload_format="incompatible",
                payload_schema_version=str(version) if version is not None else None,
                compatibility_reason="unsupported_persisted_request_schema_version",
            )
        try:
            context = LLMAnalysisContext.model_validate(context_payload)
        except (TypeError, ValueError):
            return _snapshot(
                record,
                compatibility_status="incompatible",
                payload_format="incompatible",
                payload_schema_version=version,
                compatibility_reason="invalid_compact_context",
            )
        return _snapshot(
            record,
            compatibility_status="compatible",
            payload_format="versioned_compact",
            payload_schema_version=version,
            context_schema_version=context.schema_version,
        )

    if isinstance(context_payload, dict):
        try:
            context = LLMAnalysisContext.model_validate(context_payload)
        except ValueError:
            return _snapshot(
                record,
                compatibility_status="incompatible",
                payload_format="incompatible",
                payload_schema_version=str(payload.get("schema_version") or "") or None,
                compatibility_reason="invalid_legacy_compact_context",
            )
        return _snapshot(
            record,
            compatibility_status="compatible",
            payload_format="legacy_compact_context",
            payload_schema_version=str(payload.get("schema_version") or "") or None,
            context_schema_version=context.schema_version,
        )

    if "trigger_timeframe" in payload and "supported_timeframe_states" in payload:
        try:
            request = AIReasoningRequest.model_validate(payload)
        except ValueError:
            return _snapshot(
                record,
                compatibility_status="incompatible",
                payload_format="incompatible",
                payload_schema_version=str(payload.get("schema_version") or "") or None,
                compatibility_reason="invalid_legacy_full_request",
            )
        return _snapshot(
            record,
            compatibility_status="compatible",
            payload_format="legacy_full_request",
            payload_schema_version=request.schema_version,
        )

    return _snapshot(
        record,
        compatibility_status="incompatible",
        payload_format="incompatible",
        payload_schema_version=str(payload.get("schema_version") or "") or None,
        compatibility_reason="unrecognized_persisted_request_payload",
    )
