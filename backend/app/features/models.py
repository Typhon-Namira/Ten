"""Provider-neutral, versioned feature records."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeatureRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    feature_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID
    namespace: str
    engine_name: str
    engine_version: str
    compatibility_version: str
    values: dict[str, Any]
    mode: str = "live"
    instrument: str | None = None
    timeframe: str | None = None
    effective_at: datetime | None = None
    source_event_id: str | None = None
    quality_status: str = "valid"
    payload_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("effective_at", "created_at")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("feature timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None


class FeatureSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    correlation_id: UUID
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
