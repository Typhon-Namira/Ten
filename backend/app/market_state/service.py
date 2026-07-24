"""Shadow-only Unified Market State capture and synchronization."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
import json
from typing import Any
from uuid import UUID, uuid5

from pydantic_core import to_jsonable_python

from backend.app.engines.market_data_engine.sessions import MarketSessionEngine
from backend.app.integration.models import CanonicalEventEnvelope, MarketCandlePayload

from .models import (
    REQUIRED_TIMEFRAMES,
    CapturedEngineEvidence,
    EvidenceAvailability,
    EvidenceClassification,
    EvidenceItem,
    MarketEvidenceFrame,
    MarketStateStatus,
    TimeframeState,
    UnifiedMarketState,
)
from .repository import UnifiedMarketStateRepository


_NAMESPACE = UUID("15c851a1-10a7-4b3d-a04a-a696856307cb")
_ENGINE_NAMES = (
    "market_data",
    "smc",
    "liquidity",
    "volume_profile",
    "institutional_flow",
    "market_regime",
    "economic_calendar",
)
_TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15}


def _canonical(value: Any) -> str:
    return json.dumps(to_jsonable_python(value, fallback=str), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return sha256(_canonical(value).encode()).hexdigest()


def _id(kind: str, *parts: object) -> UUID:
    return uuid5(_NAMESPACE, ":".join((kind, *(str(part) for part in parts))))


def _jsonable(value: Any) -> Any:
    return to_jsonable_python(value, fallback=str)


def _attribute(value: object, *names: str) -> Any:
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate.value if isinstance(candidate, Enum) else candidate
    return None


def _percentage(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        if 0 <= number <= 1:
            return number * 100
        if 0 <= number <= 100:
            return number
    return None


def expected_closed_boundary(boundary: datetime, timeframe: str) -> datetime:
    """Latest candle close for a timeframe that is knowable at ``boundary``."""

    if boundary.tzinfo is None:
        raise ValueError("synchronization boundary must be timezone-aware")
    minutes = _TIMEFRAME_MINUTES[timeframe]
    utc = boundary.astimezone(UTC)
    epoch_minutes = int(utc.timestamp() // 60)
    aligned_minutes = epoch_minutes - epoch_minutes % minutes
    return datetime.fromtimestamp(aligned_minutes * 60, tz=UTC)


class UnifiedMarketStateService:
    """Captures full engine outputs and builds synchronized M1/M5/M15 states."""

    def __init__(
        self,
        repository: UnifiedMarketStateRepository,
        *,
        sessions: MarketSessionEngine | None = None,
        clock: Any | None = None,
    ) -> None:
        self.repository = repository
        self.sessions = sessions or MarketSessionEngine()
        self.clock = clock or (lambda: datetime.now(UTC))

    async def capture_cycle(
        self,
        envelope: CanonicalEventEnvelope,
        outputs: dict[str, object],
    ) -> UnifiedMarketState | None:
        if not isinstance(envelope.payload, MarketCandlePayload):
            raise ValueError("Unified Market State only accepts market candle envelopes")
        timeframe = envelope.payload.timeframe
        if timeframe not in REQUIRED_TIMEFRAMES:
            return None
        frame = self._frame(envelope, outputs)
        await self.repository.save_frame(frame)
        return await self.synchronize(
            instrument=envelope.payload.canonical_instrument,
            trigger_timeframe=timeframe,
            market_data_boundary=envelope.payload.close_time,
            knowledge_cutoff=max(envelope.produced_at, self.clock()),
            cycle_id=envelope.trace_id,
            correlation_id=envelope.correlation_id,
            mode=envelope.mode.value,
        )

    async def synchronize(
        self,
        *,
        instrument: str,
        trigger_timeframe: str,
        market_data_boundary: datetime,
        knowledge_cutoff: datetime,
        cycle_id: UUID,
        correlation_id: UUID,
        mode: str,
    ) -> UnifiedMarketState | None:
        selected: dict[str, MarketEvidenceFrame] = {}
        for timeframe in REQUIRED_TIMEFRAMES:
            expected = expected_closed_boundary(market_data_boundary, timeframe)
            frame = await self.repository.latest_frame(instrument, timeframe, expected, knowledge_cutoff)
            if frame is None:
                return None
            selected[timeframe] = frame

        # UMS describes the synchronized market boundary, not the wall clock of a delayed
        # worker. Publication revalidates this schedule at its own execution timestamp.
        market_schedule = self.sessions.status_at(market_data_boundary)
        state_material = {
            "schema_version": "1.0",
            "instrument": instrument,
            "boundary": market_data_boundary.isoformat(),
            "knowledge_cutoff": knowledge_cutoff.isoformat(),
            "frames": {name: value.frame_hash for name, value in sorted(selected.items())},
            "market_schedule": market_schedule.model_dump(mode="json"),
            "mode": mode,
        }
        state_hash = _hash(state_material)
        state_id = _id("unified-market-state", state_hash)
        evidence: list[EvidenceItem] = []
        timeframe_states: list[TimeframeState] = []
        for timeframe in REQUIRED_TIMEFRAMES:
            frame = selected[timeframe]
            expected = expected_closed_boundary(market_data_boundary, timeframe)
            frame_stale = frame.candle_close_at < expected
            ids: list[UUID] = []
            for captured in frame.evidence:
                availability = EvidenceAvailability.STALE if frame_stale and captured.availability == EvidenceAvailability.AVAILABLE else captured.availability
                evidence_id = _id("evidence", state_id, frame.frame_id, captured.source_engine, _hash(captured.raw_value))
                ids.append(evidence_id)
                evidence.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        market_state_id=state_id,
                        source_frame_id=frame.frame_id,
                        source_engine=captured.source_engine,
                        source_engine_version=captured.source_engine_version,
                        source_timeframe=timeframe,
                        source_candle_timestamp=frame.candle_open_at,
                        source_candle_close_timestamp=frame.candle_close_at,
                        evidence_type=captured.evidence_type,
                        classification=captured.classification,
                        availability=availability,
                        normalized_value=captured.normalized_value,
                        raw_value=captured.raw_value,
                        confidence=captured.confidence,
                        quality=captured.quality,
                        uncertainty=captured.uncertainty,
                        observed_at=captured.observed_at,
                        available_at=captured.available_at,
                        freshness_seconds=max(0.0, (market_data_boundary - frame.candle_close_at).total_seconds()),
                        provenance=captured.provenance,
                        reason_codes=captured.reason_codes,
                    )
                )
            timeframe_states.append(
                TimeframeState(
                    timeframe=timeframe,
                    frame_id=frame.frame_id,
                    source_candle_open_at=frame.candle_open_at,
                    source_candle_close_at=frame.candle_close_at,
                    expected_candle_close_at=expected,
                    freshness_seconds=max(0.0, (market_data_boundary - frame.candle_close_at).total_seconds()),
                    stale=frame_stale,
                    evidence_ids=tuple(ids),
                )
            )

        unavailable = tuple(item.evidence_id for item in evidence if item.availability == EvidenceAvailability.UNAVAILABLE)
        degraded = tuple(item.evidence_id for item in evidence if item.availability == EvidenceAvailability.DEGRADED)
        stale = tuple(item.evidence_id for item in evidence if item.availability == EvidenceAvailability.STALE)
        available_count = sum(item.availability == EvidenceAvailability.AVAILABLE for item in evidence)
        state = UnifiedMarketState(
            state_id=state_id,
            state_hash=state_hash,
            cycle_id=cycle_id,
            correlation_id=correlation_id,
            instrument=instrument,
            trigger_timeframe=trigger_timeframe,
            market_data_boundary=market_data_boundary,
            knowledge_cutoff=knowledge_cutoff,
            mode=mode,
            status=MarketStateStatus.DEGRADED if unavailable or degraded or stale else MarketStateStatus.AVAILABLE,
            market_schedule=market_schedule,
            timeframes=tuple(timeframe_states),
            evidence=tuple(evidence),
            unavailable_evidence=unavailable,
            degraded_evidence=degraded,
            stale_evidence=stale,
            evidence_completeness=available_count / len(evidence) if evidence else 0,
            created_at=knowledge_cutoff,
        )
        return await self.repository.save_state(state)

    def _frame(self, envelope: CanonicalEventEnvelope, outputs: dict[str, object]) -> MarketEvidenceFrame:
        assert isinstance(envelope.payload, MarketCandlePayload)
        payload = envelope.payload
        knowledge_cutoff = max(envelope.produced_at, self.clock())
        captured: list[CapturedEngineEvidence] = [
            CapturedEngineEvidence(
                source_engine="market_data",
                source_engine_version=envelope.producer_version,
                classification=EvidenceClassification.MARKET_EVIDENCE,
                availability=EvidenceAvailability.AVAILABLE,
                raw_value=_jsonable(payload),
                confidence=payload.quality_score,
                quality=payload.quality_score,
                observed_at=payload.close_time,
                available_at=min(envelope.available_at, knowledge_cutoff),
                provenance={
                    "provider": payload.provider,
                    "provider_symbol": payload.provider_symbol,
                    "market_event_id": envelope.event_id,
                },
            )
        ]
        for engine in _ENGINE_NAMES[1:]:
            value = outputs.get(engine)
            if value is None:
                captured.append(
                    CapturedEngineEvidence(
                        source_engine=engine,
                        source_engine_version="unavailable",
                        availability=EvidenceAvailability.UNAVAILABLE,
                        observed_at=payload.close_time,
                        available_at=knowledge_cutoff,
                        provenance={"market_event_id": envelope.event_id},
                        reason_codes=("engine_output_unavailable",),
                    )
                )
                continue
            status = str(_attribute(value, "status") or "").lower()
            availability = EvidenceAvailability.DEGRADED if any(token in status for token in ("degrad", "insufficient", "unavailable", "error")) else EvidenceAvailability.AVAILABLE
            observed = _attribute(value, "analysis_timestamp", "as_of", "historical_boundary") or payload.close_time
            if isinstance(observed, datetime) and observed > knowledge_cutoff:
                raise ValueError(f"future engine evidence: {engine}")
            if not isinstance(observed, datetime):
                observed = payload.close_time
            available = _attribute(value, "created_at", "calculated_at", "available_at") or knowledge_cutoff
            if isinstance(available, datetime) and available > knowledge_cutoff:
                raise ValueError(f"future engine availability: {engine}")
            if not isinstance(available, datetime):
                available = knowledge_cutoff
            captured.append(
                CapturedEngineEvidence(
                    source_engine=engine,
                    source_engine_version=str(_attribute(value, "engine_version") or "1.0.0"),
                    availability=availability,
                    raw_value=_jsonable(value),
                    confidence=_percentage(_attribute(value, "confidence_score", "confidence")),
                    quality=_percentage(_attribute(value, "quality_score", "quality")),
                    observed_at=observed,
                    available_at=available,
                    provenance={
                        "snapshot_id": str(_attribute(value, "snapshot_id", "id", "context_id") or ""),
                        "market_event_id": envelope.event_id,
                    },
                    reason_codes=("engine_reported_degraded",) if availability == EvidenceAvailability.DEGRADED else (),
                )
            )
        material = {
            "event_id": envelope.event_id,
            "timeframe": payload.timeframe,
            "candle_close": payload.close_time.isoformat(),
            "evidence": [item.model_dump(mode="json") for item in captured],
        }
        frame_hash = _hash(material)
        return MarketEvidenceFrame(
            frame_id=_id("evidence-frame", frame_hash),
            frame_hash=frame_hash,
            cycle_id=envelope.trace_id,
            correlation_id=envelope.correlation_id,
            instrument=payload.canonical_instrument,
            timeframe=payload.timeframe,
            candle_open_at=payload.open_time,
            candle_close_at=payload.close_time,
            knowledge_cutoff=knowledge_cutoff,
            mode=envelope.mode.value,
            market_event_id=envelope.event_id,
            evidence=tuple(captured),
            created_at=knowledge_cutoff,
        )
