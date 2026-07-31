"""Durable authoritative AI gate diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AIReasoningGateDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: UUID
    instrument: str
    trigger_timeframe: str | None = None
    attempted_cutoff: datetime
    analysis_lookup_cutoff: datetime
    market_state_id: UUID | None = None
    snapshot_id: UUID | None = None
    gate_decision: str
    gate_skip_reason: str | None = None
    existing_analysis_id: UUID | None = None
    analysis_created_at: datetime | None = None
    analysis_market_cutoff: datetime | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @field_validator(
        "attempted_cutoff",
        "analysis_lookup_cutoff",
        "analysis_created_at",
        "analysis_market_cutoff",
        "created_at",
    )
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("AI gate timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None
