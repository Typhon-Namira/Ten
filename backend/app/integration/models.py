from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Literal
from uuid import UUID, NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.engines.market_data_engine import Candle


JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


def canonical_hash(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def semantic_uuid(*parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, "ten:integration:" + "|".join(str(part) for part in parts))


class IntegrationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IntegrationMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"


class DataQualityStatus(StrEnum):
    VALID = "valid"
    SUSPECT = "suspect"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"


class SnapshotStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    INSUFFICIENT_DATA = "insufficient_data"


class TraceStatus(StrEnum):
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class MarketCandlePayload(IntegrationModel):
    payload_kind: Literal["market_candle"] = "market_candle"
    symbol: str
    canonical_instrument: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    ingestion_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0)
    spread: float = Field(ge=0)
    provider: str
    provider_symbol: str
    quality_score: float = Field(ge=0, le=100)
    quality_level: str
    final: bool = True
    revision: int = Field(default=1, ge=1)

    @field_validator("open_time", "close_time", "ingestion_time")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("candle timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def candle_invariants(self) -> MarketCandlePayload:
        if self.close_time <= self.open_time:
            raise ValueError("candle close must follow open")
        if not all(math.isfinite(value) and value > 0 for value in (self.open, self.high, self.low, self.close)):
            raise ValueError("OHLC values must be finite and positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close) or self.low > self.high:
            raise ValueError("invalid OHLC relationship")
        if not math.isfinite(self.volume) or not math.isfinite(self.spread):
            raise ValueError("volume and spread must be finite")
        return self

    @classmethod
    def from_candle(cls, candle: Candle) -> MarketCandlePayload:
        return cls(
            symbol=candle.symbol,
            canonical_instrument=candle.symbol.upper().replace("/", "").replace("-", "").replace("_", ""),
            timeframe=candle.timeframe.value,
            open_time=candle.timestamp,
            close_time=candle.timestamp + candle.timeframe.duration,
            ingestion_time=candle.ingestion_timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            spread=candle.spread,
            provider=candle.provider,
            provider_symbol=candle.symbol,
            quality_score=candle.quality_score,
            quality_level=candle.quality_level.value,
        )


class SnapshotReadyPayload(IntegrationModel):
    payload_kind: Literal["snapshot_ready"] = "snapshot_ready"
    snapshot_id: UUID
    evidence_hash: str = Field(min_length=64, max_length=64)


class AIScoreGeneratedPayload(IntegrationModel):
    payload_kind: Literal["ai_score_generated"] = "ai_score_generated"
    score_id: UUID
    snapshot_id: UUID
    semantic_hash: str = Field(min_length=64, max_length=64)


class SignalDecisionGeneratedPayload(IntegrationModel):
    payload_kind: Literal["signal_decision_generated"] = "signal_decision_generated"
    decision_id: UUID
    snapshot_id: UUID
    score_id: UUID
    semantic_hash: str = Field(min_length=64, max_length=64)


CanonicalPayload = MarketCandlePayload | SnapshotReadyPayload | AIScoreGeneratedPayload | SignalDecisionGeneratedPayload


class CanonicalEventEnvelope(IntegrationModel):
    event_id: str = Field(min_length=64, max_length=64)
    event_type: str = Field(pattern=r"^[a-z][a-z0-9_.]+$")
    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    producer_engine: str
    producer_version: str
    trace_id: UUID
    correlation_id: UUID
    causation_id: str | None = None
    instrument_id: str | None = None
    timeframe: str | None = None
    mode: IntegrationMode
    occurred_at: datetime
    available_at: datetime
    produced_at: datetime
    source_name: str
    source_event_id: str | None = None
    source_version: str | None = None
    data_quality_status: DataQualityStatus
    payload_hash: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=64, max_length=64)
    payload: CanonicalPayload

    @field_validator("occurred_at", "available_at", "produced_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def invariants(self) -> CanonicalEventEnvelope:
        if self.available_at < self.occurred_at or self.produced_at < self.available_at:
            raise ValueError("event availability order is invalid")
        if canonical_hash(self.payload) != self.payload_hash:
            raise ValueError("event payload hash mismatch")
        if self.mode == IntegrationMode.REPLAY and self.source_name == "live_provider":
            raise ValueError("Replay cannot consume a live-provider envelope")
        return self

    @classmethod
    def final_candle(cls, candle: Candle, correlation_id: UUID, produced_at: datetime) -> CanonicalEventEnvelope:
        payload = MarketCandlePayload.from_candle(candle)
        payload_hash = canonical_hash(payload)
        source_event_id = canonical_hash(
            {"provider": candle.provider, "symbol": payload.canonical_instrument, "timeframe": payload.timeframe, "open_time": payload.open_time.isoformat(), "revision": payload.revision}
        )
        # Deliberately derived from `source_event_id` alone, NOT `payload_hash` — `payload_hash`
        # is a hash of the WHOLE payload, which includes `ingestion_time`, a field stamped fresh
        # (`datetime.now(UTC)`) on every single fetch, live or replayed. Folding it into `event_id`
        # meant two polls of the exact same already-closed candle, seconds apart, produced two
        # different `event_id`s — `repository.processed(event_id)` could never recognize the second
        # poll as a duplicate, so `process()` re-ran the entire SMC-onward chain from scratch on
        # every single poll of an already-processed candle, forever. `source_event_id` already
        # captures everything that makes a candle a *new* event (provider, instrument, timeframe,
        # open time, revision) — a later poll of the same bar must hash to the same `event_id`.
        event_id = canonical_hash({"type": "market.candle.closed", "source_event_id": source_event_id, "schema": "1.0"})
        trace_id = semantic_uuid("trace", event_id)
        quality = DataQualityStatus.VALID if candle.quality_score >= 80 else DataQualityStatus.SUSPECT if candle.quality_score >= 50 else DataQualityStatus.REJECTED
        return cls(
            event_id=event_id,
            event_type="market.candle.closed",
            schema_version="1.0",
            producer_engine="market_data",
            producer_version="1.0.0",
            trace_id=trace_id,
            correlation_id=correlation_id,
            instrument_id=payload.canonical_instrument,
            timeframe=payload.timeframe,
            mode=IntegrationMode.LIVE,
            occurred_at=payload.open_time,
            available_at=max(payload.close_time, payload.ingestion_time),
            produced_at=produced_at,
            source_name=candle.provider,
            source_event_id=source_event_id,
            source_version="1.0.0",
            data_quality_status=quality,
            payload_hash=payload_hash,
            idempotency_key=event_id,
            payload=payload,
        )

    @classmethod
    def historical_candle(cls, candle: Candle, correlation_id: UUID, produced_at: datetime) -> CanonicalEventEnvelope:
        """Create a mode-isolated envelope for bootstrap analysis, never live publication."""
        live = cls.final_candle(candle, correlation_id, max(produced_at, candle.timestamp + candle.timeframe.duration, candle.ingestion_timestamp))
        close_time = candle.timestamp + candle.timeframe.duration
        event_id = canonical_hash({"live_event_id": live.event_id, "mode": IntegrationMode.REPLAY.value, "purpose": "historical_bootstrap"})
        return live.model_copy(
            update={
                "event_id": event_id,
                "trace_id": semantic_uuid("trace", event_id),
                "mode": IntegrationMode.REPLAY,
                "available_at": close_time,
                "produced_at": max(produced_at, close_time),
                "idempotency_key": event_id,
            }
        )


class EvidenceReference(IntegrationModel):
    engine: str
    evidence_id: str
    engine_version: str
    effective_at: datetime
    required: bool = True
    quality_status: DataQualityStatus = DataQualityStatus.VALID

    @field_validator("effective_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("evidence timestamp must be timezone-aware")
        return value.astimezone(UTC)


class IntegrationSnapshot(IntegrationModel):
    snapshot_id: UUID
    semantic_hash: str = Field(min_length=64, max_length=64)
    mode: IntegrationMode
    instrument: str
    timeframe: str
    analytical_boundary: datetime
    market_event_id: str
    evidence: tuple[EvidenceReference, ...]
    missing_required: tuple[str, ...] = ()
    stale_evidence: tuple[str, ...] = ()
    data_quality_status: DataQualityStatus
    status: SnapshotStatus
    created_at: datetime

    @field_validator("analytical_boundary", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("snapshot timestamps must be timezone-aware")
        return value.astimezone(UTC)


class OperationalSignal(IntegrationModel):
    operational_signal_id: UUID
    semantic_hash: str = Field(min_length=64, max_length=64)
    decision_id: UUID
    ai_score_id: UUID
    snapshot_id: UUID
    trace_id: UUID
    market_event_id: str
    instrument: str
    timeframe: str
    mode: IntegrationMode
    direction: str
    state: str
    confidence: float = Field(ge=0, le=100)
    effective_at: datetime
    expires_at: datetime
    data_quality_status: DataQualityStatus
    provider_provenance: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    ai_scoring_policy_version: str
    signal_decision_policy_version: str
    created_at: datetime
    analytical_only: bool = True
    trade_execution: bool = False

    @field_validator("effective_at", "expires_at", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("signal timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def safety(self) -> OperationalSignal:
        if not self.analytical_only or self.trade_execution:
            raise ValueError("operational signals are analytical only")
        return self


class IntegrationTraceRecord(IntegrationModel):
    trace_record_id: UUID
    trace_id: UUID
    event_id: str
    event_type: str
    producer: str
    consumer: str
    instrument: str | None = None
    timeframe: str | None = None
    mode: IntegrationMode
    status: TraceStatus
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0)
    input_references: tuple[str, ...] = ()
    output_references: tuple[str, ...] = ()
    error_code: str | None = None
    retry_count: int = Field(default=0, ge=0)
    correlation_id: UUID
    causation_id: str | None = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("trace timestamps must be timezone-aware")
        return value.astimezone(UTC)


class DataQualityIssue(IntegrationModel):
    issue_id: UUID
    event_id: str
    instrument: str
    timeframe: str
    status: DataQualityStatus
    reason_codes: tuple[str, ...]
    observed_at: datetime
    provider: str

    @field_validator("observed_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("quality timestamp must be timezone-aware")
        return value.astimezone(UTC)


class OutboxItem(IntegrationModel):
    outbox_id: UUID
    envelope: CanonicalEventEnvelope
    available_at: datetime
    attempts: int = Field(default=0, ge=0)
    published_at: datetime | None = None
    last_error_code: str | None = None

    @field_validator("available_at", "published_at")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("outbox timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None
