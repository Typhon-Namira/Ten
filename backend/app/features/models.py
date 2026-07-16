"""Provider-neutral, versioned feature records."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FeatureRecord(BaseModel):
    feature_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID
    namespace: str
    engine_name: str
    engine_version: str
    compatibility_version: str
    values: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FeatureSnapshot(BaseModel):
    correlation_id: UUID
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
