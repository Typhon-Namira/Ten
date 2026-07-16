"""Shared runtime contracts between discovered engines and the pipeline manager."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from backend.app.engines.economic_calendar_engine import EconomicEvent
from backend.app.engines.market_data_engine import Candle
from backend.app.events import Event
from backend.app.features import FeatureSnapshot, FeatureStore


@dataclass
class PipelineExecutionContext:
    correlation_id: UUID
    candles: list[Candle]
    events: list[EconomicEvent]
    feature_store: FeatureStore
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    results: dict[str, Any] = field(default_factory=dict)
    feature_snapshot: FeatureSnapshot | None = None
    calculated_confidence: float | None = None
    confidence_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineExecutionResult:
    output: Any
    features: dict[str, Any]
    namespace: str
    event_type: type[Event]
    confidence_factor: float | None = None
