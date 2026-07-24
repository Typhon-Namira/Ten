from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import BaseModel

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.integration import CanonicalEventEnvelope
from backend.app.market_state import (
    EvidenceAvailability,
    InMemoryUnifiedMarketStateRepository,
    UnifiedMarketStateService,
    expected_closed_boundary,
)
from backend.app.market_state.repository import (
    _compact_evidence_payload,
    _compact_state_payload,
    _reconstruct_compact_state,
)


BOUNDARY = datetime(2026, 7, 23, 12, 30, tzinfo=UTC)
KNOWLEDGE = BOUNDARY + timedelta(seconds=3)


class RichEngineOutput(BaseModel):
    snapshot_id: str
    analysis_timestamp: datetime
    created_at: datetime
    engine_version: str = "1.0.0"
    status: str = "ready"
    confidence_score: float = 87.5
    quality_score: float = 93.0
    structure: dict[str, object]
    objects: list[dict[str, object]]


def candle(timeframe: Timeframe, close_at: datetime = BOUNDARY) -> Candle:
    return Candle(
        timestamp=close_at - timeframe.duration,
        ingestion_timestamp=close_at,
        symbol="XAU/USD",
        timeframe=timeframe,
        open=3300,
        high=3305,
        low=3298,
        close=3302,
        volume=100,
        spread=0.2,
        provider="test-provider",
    )


def output(engine: str, observed_at: datetime = BOUNDARY) -> RichEngineOutput:
    return RichEngineOutput(
        snapshot_id=f"{engine}-snapshot",
        analysis_timestamp=observed_at,
        created_at=KNOWLEDGE,
        structure={
            "internal": {"direction": "bullish", "protected_low": 3298.25},
            "external": {"direction": "bearish", "protected_high": 3312.75},
        },
        objects=[
            {"kind": "order_block", "bounds": {"low": 3299.1, "high": 3301.4}},
            {"kind": "fvg", "bounds": {"low": 3302.2, "high": 3303.8}},
        ],
    )


def outputs() -> dict[str, object]:
    return {
        name: output(name)
        for name in (
            "smc",
            "liquidity",
            "volume_profile",
            "institutional_flow",
            "market_regime",
            "economic_calendar",
        )
    }


async def capture(service: UnifiedMarketStateService, timeframe: Timeframe, values: dict[str, object] | None = None, close_at: datetime = BOUNDARY):
    envelope = CanonicalEventEnvelope.final_candle(candle(timeframe, close_at), uuid4(), max(KNOWLEDGE, close_at + timedelta(seconds=3)))
    return await service.capture_cycle(envelope, outputs() if values is None else values)


@pytest.mark.asyncio
async def test_m1_m5_m15_are_synchronized_at_the_same_point_in_time() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE)

    assert await capture(service, Timeframe.M1) is None
    assert await capture(service, Timeframe.M5) is None
    state = await capture(service, Timeframe.M15)

    assert state is not None
    assert tuple(item.timeframe for item in state.timeframes) == ("M1", "M5", "M15")
    assert all(item.source_candle_close_at == BOUNDARY for item in state.timeframes)
    assert all(item.expected_candle_close_at == BOUNDARY for item in state.timeframes)
    assert all(not item.stale for item in state.timeframes)
    assert len(state.evidence) == 21
    assert state.evidence_completeness == 1


@pytest.mark.asyncio
async def test_synchronizer_never_selects_a_future_timeframe_frame() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE + timedelta(minutes=5))
    future_close = BOUNDARY + timedelta(minutes=5)

    await capture(service, Timeframe.M1, close_at=BOUNDARY)
    await capture(service, Timeframe.M5, close_at=future_close)
    await capture(service, Timeframe.M15, close_at=BOUNDARY)

    state = await service.synchronize(
        instrument="XAUUSD",
        trigger_timeframe="M15",
        market_data_boundary=BOUNDARY,
        knowledge_cutoff=KNOWLEDGE + timedelta(minutes=5),
        cycle_id=uuid4(),
        correlation_id=uuid4(),
        mode="live",
    )
    assert state is None


@pytest.mark.asyncio
async def test_capture_rejects_future_engine_data_leakage() -> None:
    service = UnifiedMarketStateService(InMemoryUnifiedMarketStateRepository(), clock=lambda: KNOWLEDGE)
    future = outputs()
    future["smc"] = output("smc", KNOWLEDGE + timedelta(seconds=1))
    with pytest.raises(ValueError, match="future engine evidence"):
        await capture(service, Timeframe.M1, future)


@pytest.mark.asyncio
async def test_complete_nested_engine_output_is_preserved_without_scalar_collapse() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE)
    await capture(service, Timeframe.M1)
    await capture(service, Timeframe.M5)
    state = await capture(service, Timeframe.M15)
    assert state is not None

    smc = next(item for item in state.evidence if item.source_engine == "smc" and item.source_timeframe == "M15")
    assert smc.raw_value["structure"]["internal"]["protected_low"] == 3298.25
    assert smc.raw_value["structure"]["external"]["protected_high"] == 3312.75
    assert smc.raw_value["objects"][0]["kind"] == "order_block"
    assert smc.raw_value["objects"][1]["bounds"]["high"] == 3303.8
    assert smc.normalized_value is None
    for engine in ("smc", "liquidity", "volume_profile", "institutional_flow", "market_regime", "economic_calendar"):
        item = next(value for value in state.evidence if value.source_engine == engine and value.source_timeframe == "M15")
        assert item.raw_value["structure"]["internal"]["direction"] == "bullish"
        assert len(item.raw_value["objects"]) == 2


@pytest.mark.asyncio
async def test_compact_persistence_owns_large_payload_once_without_losing_runtime_evidence() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE)
    await capture(service, Timeframe.M1)
    await capture(service, Timeframe.M5)
    state = await capture(service, Timeframe.M15)
    assert state is not None

    full_state = json.dumps(state.model_dump(mode="json"))
    compact_state = json.dumps(_compact_state_payload(state))
    compact_evidence = json.dumps(
        [_compact_evidence_payload(item) for item in state.evidence]
    )

    assert '"raw_value"' not in compact_state
    assert '"evidence"' not in compact_state
    assert '"raw_value"' not in compact_evidence
    assert len(compact_state) + len(compact_evidence) < len(full_state) * 0.50
    # Compaction is persistence-only. The in-flight analytical state is byte-for-byte untouched.
    smc = next(item for item in state.evidence if item.source_engine == "smc")
    assert smc.raw_value["objects"][0]["kind"] == "order_block"


@pytest.mark.asyncio
async def test_legacy_and_compact_rows_reconstruct_semantically_identical_state() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE)
    await capture(service, Timeframe.M1)
    await capture(service, Timeframe.M5)
    state = await capture(service, Timeframe.M15)
    assert state is not None

    legacy = type(state).model_validate(state.model_dump(mode="json"))
    record = SimpleNamespace(
        payload=_compact_state_payload(state),
        market_data_boundary=state.market_data_boundary,
    )
    timeframe_rows = [
        SimpleNamespace(
            frame_id=item.frame_id,
            timeframe=item.timeframe,
            source_candle_close_at=item.source_candle_close_at,
            expected_candle_close_at=item.expected_candle_close_at,
            stale=item.stale,
        )
        for item in state.timeframes
    ]
    evidence_rows = [
        (
            SimpleNamespace(ordinal=ordinal),
            SimpleNamespace(
                evidence_id=item.evidence_id,
                source_frame_id=item.source_frame_id,
                source_engine=item.source_engine,
                payload=_compact_evidence_payload(item),
            ),
        )
        for ordinal, item in enumerate(state.evidence)
    ]
    compact = _reconstruct_compact_state(
        record,
        timeframe_rows,
        dict(repository._frames),
        evidence_rows,
    )

    assert compact.model_dump(mode="json") == legacy.model_dump(mode="json")
    assert compact.state_hash == legacy.state_hash


@pytest.mark.asyncio
async def test_repeated_unchanged_cycles_do_not_grow_frames_or_state_history() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE)
    for _ in range(25):
        await capture(service, Timeframe.M1)
        await capture(service, Timeframe.M5)
        await capture(service, Timeframe.M15)

    assert len(repository._frames) == 3
    assert len(repository._states) == 1
    state = next(iter(repository._states.values()))
    assert len(state.evidence) == 21


@pytest.mark.asyncio
async def test_unavailable_evidence_is_explicit_and_never_serialized_as_zero() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE)
    missing_volume = outputs()
    missing_volume.pop("volume_profile")

    await capture(service, Timeframe.M1, missing_volume)
    await capture(service, Timeframe.M5, missing_volume)
    state = await capture(service, Timeframe.M15, missing_volume)
    assert state is not None

    unavailable = [item for item in state.evidence if item.source_engine == "volume_profile"]
    assert len(unavailable) == 3
    assert all(item.availability == EvidenceAvailability.UNAVAILABLE for item in unavailable)
    assert all(item.raw_value is None and item.normalized_value is None for item in unavailable)
    assert all(item.confidence is None and item.quality is None and item.uncertainty is None for item in unavailable)
    assert set(state.unavailable_evidence) == {item.evidence_id for item in unavailable}
    assert state.evidence_completeness == pytest.approx(18 / 21)


@pytest.mark.asyncio
async def test_older_closed_frame_is_retained_but_explicitly_marked_stale() -> None:
    repository = InMemoryUnifiedMarketStateRepository()
    service = UnifiedMarketStateService(repository, clock=lambda: KNOWLEDGE)
    await capture(service, Timeframe.M1, close_at=BOUNDARY)
    await capture(service, Timeframe.M5, close_at=BOUNDARY)
    await capture(service, Timeframe.M15, close_at=BOUNDARY - timedelta(minutes=15))

    state = await service.synchronize(
        instrument="XAUUSD",
        trigger_timeframe="M1",
        market_data_boundary=BOUNDARY,
        knowledge_cutoff=KNOWLEDGE,
        cycle_id=uuid4(),
        correlation_id=uuid4(),
        mode="live",
    )
    assert state is not None
    m15 = next(item for item in state.timeframes if item.timeframe == "M15")
    assert m15.stale
    assert m15.expected_candle_close_at == expected_closed_boundary(BOUNDARY, "M15")
    assert all(
        item.availability == EvidenceAvailability.STALE
        for item in state.evidence
        if item.source_timeframe == "M15"
    )
