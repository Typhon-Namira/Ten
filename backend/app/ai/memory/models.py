from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SignalMemoryRecord(BaseModel):
    record_id: UUID = Field(default_factory=uuid4)
    signal_id: str
    features: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OutcomeRecord(BaseModel):
    signal_id: str
    outcome: str
    metrics: dict[str, float] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReasoningRecord(BaseModel):
    signal_id: str
    model: str
    prompt_version: str
    reasoning: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryContext(BaseModel):
    query: str
    records: list[dict[str, Any]] = Field(default_factory=list)
