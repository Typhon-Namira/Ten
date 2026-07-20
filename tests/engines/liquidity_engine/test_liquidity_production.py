from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.liquidity_engine import BaselineLiquidityAnalyzer, LiquidityConfig, LiquidityContext, LiquidityService
from backend.app.engines.liquidity_engine.config import (
    EqualLevelConfig,
    MultiTimeframeConfig,
    PersistenceConfig,
    PoolConfig,
    RoundNumberConfig,
    ToleranceConfig,
)
from backend.app.engines.liquidity_engine.models import (
    AnalysisStatus,
    LiquidityEvidence,
    LiquidityLevel,
    LiquidityLifecycleState,
    LiquidityPool,
    LiquidityScope,
    ProcessingMode,
    utc,
    validate_transition,
    _TRANSITIONS,
)
from backend.app.engines.liquidity_engine.registration import _build, _execute
from backend.app.engines.liquidity_engine.registration import register
from backend.app.engines.liquidity_engine.repository import InMemoryLiquidityRepository, SqlAlchemyLiquidityRepository
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.smc_engine.liquidity_contract import SMCLiquidityContext, SMCLiquidityLevel
from tests.conftest import FakeSessionFactory
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.services.engine_factory import EngineBuildContext
from backend.app.services.engine_factory import EngineFactory
from backend.app.services.pipeline_contracts import PipelineExecutionContext


def series(
    rows: list[tuple[float, float, float, float]], *, start: datetime | None = None, quality: float = 100, timeframe: Timeframe = Timeframe.M15
) -> list[Candle]:
    origin = start or datetime(2026, 1, 1, 8, tzinfo=UTC)
    return [
        Candle(
            symbol="XAU/USD",
            timeframe=timeframe,
            timestamp=origin + timeframe.duration * index,
            open=o,
            high=h,
            low=low,
            close=c,
            volume=100 + index,
            quality_score=quality,
        )
        for index, (o, h, low, c) in enumerate(rows)
    ]


def smc_context(candles: list[Candle], *, future: bool = False) -> SMCLiquidityContext:
    at = candles[-1].timestamp + (timedelta(days=1) if future else timedelta(0))
    levels = (
        SMCLiquidityLevel(
            id="h1",
            symbol="XAUUSD",
            timeframe=candles[-1].timeframe,
            kind="external_high",
            scope="external",
            price=101,
            occurred_at=candles[0].timestamp,
            available_at=candles[1].timestamp,
            confidence_score=90,
            quality_score=95,
        ),
        SMCLiquidityLevel(
            id="h2",
            symbol="XAUUSD",
            timeframe=candles[-1].timeframe,
            kind="external_high",
            scope="external",
            price=101.01,
            occurred_at=candles[1].timestamp,
            available_at=candles[2].timestamp,
            confidence_score=85,
            quality_score=90,
        ),
        SMCLiquidityLevel(
            id="future",
            symbol="XAUUSD",
            timeframe=candles[-1].timeframe,
            kind="swing_low",
            scope="internal",
            price=95,
            occurred_at=candles[-1].timestamp,
            available_at=at,
            confidence_score=75,
            quality_score=80,
        ),
    )
    return SMCLiquidityContext(
        symbol="XAUUSD",
        timeframe=candles[-1].timeframe,
        analyzed_through=candles[-1].timestamp,
        structure_direction="bullish",
        levels=levels,
        protected_level_ids=("h1",),
        structural_event_ids=("bos",),
        configuration_version="smc-config",
        engine_version="3.0.0",
    )


def test_domain_invariants_configuration_and_transitions() -> None:
    config = LiquidityConfig()
    assert config.version == LiquidityConfig().version and config.max_levels == config.pools.maximum_active
    assert config.equal_level_tolerance == config.tolerances.percentage
    with pytest.raises(ValidationError, match="weights"):
        LiquidityConfig.model_validate({"ranking": {"distance_weight": 1}})
    with pytest.raises(ValidationError, match="tolerance"):
        LiquidityConfig(tolerances=ToleranceConfig(absolute=0, ticks=0, atr_multiplier=0, percentage=0))
    assert LiquidityEvidence(code="bounded", weight=1).model_dump()["passed"] is True
    with pytest.raises(ValidationError):
        LiquidityEvidence(code="bad", weight=2)
    for previous, allowed in _TRANSITIONS.items():
        validate_transition(previous, previous)
        for current in allowed:
            validate_transition(previous, current)
    with pytest.raises(ValueError, match="impossible"):
        validate_transition(LiquidityLifecycleState.CONSUMED, LiquidityLifecycleState.ACTIVE)
    aware = datetime.now(UTC)
    assert utc(aware) == aware
    with pytest.raises(ValueError, match="timezone"):
        utc(datetime.now())


def test_empty_insufficient_disabled_and_stable_prefix() -> None:
    analyzer = BaselineLiquidityAnalyzer()
    empty = analyzer.analyze([])
    assert empty.snapshot and empty.snapshot.status == AnalysisStatus.INSUFFICIENT_HISTORY
    short = series([(100, 101, 99, 100)])
    assert analyzer.analyze(short).observations
    candles = series([(100, 101, 99, 100), (100, 101.01, 99, 100), (100, 102, 98, 101), (101, 103, 100, 102), (102, 104, 101, 103)])
    config = LiquidityConfig(equal_levels=EqualLevelConfig(enabled=False), round_numbers=RoundNumberConfig(enabled=False))
    snapshot = BaselineLiquidityAnalyzer(config).analyze_snapshot(LiquidityContext(tuple(candles)))
    assert not snapshot.equal_levels and all(item.level_type.value != "round_number" for item in snapshot.levels)
    prefix = analyzer.analyze_snapshot(LiquidityContext(tuple(candles)))
    extended = analyzer.analyze_snapshot(LiquidityContext(tuple(candles + series([(103, 110, 90, 105)], start=candles[-1].timestamp + Timeframe.M15.duration))))
    replay = analyzer.analyze_snapshot(LiquidityContext(tuple(candles)), ProcessingMode.REPLAY)
    incremental = analyzer.analyze_snapshot(LiquidityContext(tuple(candles)), ProcessingMode.INCREMENTAL)
    assert prefix.analysis_timestamp == replay.analysis_timestamp < extended.analysis_timestamp
    assert {item.id for item in prefix.equal_levels} == {item.id for item in replay.equal_levels}
    assert prefix.levels == incremental.levels and prefix.pools == incremental.pools and prefix.targets == incremental.targets


def test_level_and_pool_temporal_range_invariants() -> None:
    candles = series([(100, 101, 99, 100), (100, 101.01, 99, 100), (100, 102, 98, 101), (101, 103, 99, 102), (102, 104, 100, 103)])
    snapshot = BaselineLiquidityAnalyzer().analyze_snapshot(LiquidityContext(tuple(candles)))
    level = snapshot.levels[0]
    data = level.model_dump()
    with pytest.raises(ValidationError, match="within"):
        LiquidityLevel.model_validate({**data, "price": data["upper_bound"] + 1})
    with pytest.raises(ValidationError, match="analysis boundary"):
        LiquidityLevel.model_validate({**data, "available_at": candles[-1].timestamp + timedelta(days=1)})
    with pytest.raises(ValidationError, match="source observations"):
        LiquidityLevel.model_validate({**data, "available_at": candles[0].timestamp, "source_timestamps": (candles[-1].timestamp,)})
    empty_sources = LiquidityLevel.model_validate({**data, "source_timestamps": ()})
    assert empty_sources.last_seen == empty_sources.available_at and level.touches >= 1
    swept = level.model_copy(update={"lifecycle_state": LiquidityLifecycleState.SWEPT})
    assert swept.swept is True
    pool = snapshot.pools[0]
    pool_data = pool.model_dump()
    with pytest.raises(ValidationError, match="upper"):
        LiquidityPool.model_validate({**pool_data, "lower_bound": pool.upper_bound + 1})
    with pytest.raises(ValidationError, match="earlier"):
        LiquidityPool.model_validate({**pool_data, "available_at": candles[-1].timestamp + timedelta(days=1)})


def test_smc_scope_tolerances_outliers_references_rounds_and_sessions() -> None:
    start = datetime(2026, 1, 30, 8, tzinfo=UTC)
    candles = series([(100, 101, 99, 100), (100, 101.01, 98.5, 100.5), (100.5, 101.005, 98, 100), (100, 102, 97, 101), (101, 103, 96, 102)], start=start)
    snap = BaselineLiquidityAnalyzer().analyze_snapshot(LiquidityContext(tuple(candles), smc_context(candles, future=True)))
    assert any(item.scope == LiquidityScope.EXTERNAL for item in snap.equal_levels)
    assert all(item.available_at <= snap.analysis_timestamp for item in snap.levels)
    assert "future" not in {source for item in snap.levels for source in item.source_object_ids}
    assert any(item.level_type.value == "round_number" for item in snap.levels)
    assert snap.sessions and snap.sessions[-1].high >= snap.sessions[-1].low
    multi_period = series([(100, 101, 99, 100), (100, 103, 98, 102)], start=datetime(2025, 12, 31, 8, tzinfo=UTC), timeframe=Timeframe.D1)
    multi_period += series(
        [(102, 104, 97, 103), (103, 105, 96, 104), (104, 106, 95, 105), (105, 107, 94, 106), (106, 108, 93, 107)],
        start=datetime(2026, 1, 2, 8, tzinfo=UTC),
        timeframe=Timeframe.D1,
    )
    period = BaselineLiquidityAnalyzer().analyze_snapshot(LiquidityContext(tuple(multi_period)))
    kinds = {item.level_type.value for item in period.reference_levels}
    assert {"previous_day_high", "previous_day_low"} <= kinds
    assert all(item.available_at <= period.analysis_timestamp for item in period.reference_levels)


def test_outlier_rejection_round_distance_and_closed_session_paths() -> None:
    candles = series([(100, 100.0, 99, 99.5), (99.5, 100.05, 99, 99.5), (99.5, 100.1, 99, 99.5), (99.5, 102, 98, 100), (100, 103, 97, 101)])
    harsh = LiquidityConfig(tolerances=ToleranceConfig(absolute=1), equal_levels=EqualLevelConfig(outlier_zscore=0.01))
    assert not BaselineLiquidityAnalyzer(harsh).analyze_snapshot(LiquidityContext(tuple(candles))).equal_levels
    tiny = series([(2, 2.1, 1.9, 2), (2, 2.1, 1.9, 2), (2, 2.2, 1.8, 2), (2, 2.3, 1.7, 2), (2, 2.4, 1.6, 2)], start=datetime(2026, 1, 3, 18, tzinfo=UTC))
    config = LiquidityConfig(round_numbers=RoundNumberConfig(maximum_distance_atr=0.01))
    snapshot = BaselineLiquidityAnalyzer(config).analyze_snapshot(LiquidityContext(tuple(tiny)))
    assert not snapshot.sessions


def test_sweep_grab_stop_hunt_false_break_raid_and_consumption() -> None:
    wick = series([(100, 101, 99, 100), (100, 101.01, 99, 100), (100, 100.5, 98, 99), (99, 103, 98, 100), (100, 100.5, 97, 99)])
    snap = BaselineLiquidityAnalyzer().analyze_snapshot(LiquidityContext(tuple(wick), smc_context(wick)))
    assert snap.sweeps and snap.grabs and snap.stop_hunts
    assert all(item.reclaim_timestamp <= snap.analysis_timestamp for item in snap.sweeps if item.reclaim_timestamp)
    close_through = series([(100, 101, 99, 100), (100, 101.01, 99, 100), (100, 102, 99, 101.5), (101.5, 102, 98, 100), (100, 101, 97, 99)])
    consumed = BaselineLiquidityAnalyzer().analyze_snapshot(LiquidityContext(tuple(close_through), smc_context(close_through)))
    assert any(item.lifecycle_state in {LiquidityLifecycleState.CONSUMED, LiquidityLifecycleState.RECLAIMED} for item in consumed.pools)
    assert consumed.false_breaks
    assert all(item.price_action_classification_only for item in snap.stop_hunts)
    assert len({item.id for item in snap.events}) == len(snap.events)


def test_expiration_and_approach_lifecycle_states() -> None:
    candles = series([(100, 100.05, 99.95, 100)] * 12)
    base = BaselineLiquidityAnalyzer().analyze_snapshot(LiquidityContext(tuple(candles))).pools[0]
    expiring = base.model_copy(update={"available_at": candles[0].timestamp, "lower_bound": 200, "upper_bound": 201})
    expired = BaselineLiquidityAnalyzer(LiquidityConfig(pools=PoolConfig(expiration_candles=10)))._lifecycle([expiring], candles, 0.1)[-1][0]
    approaching = base.model_copy(update={"available_at": candles[0].timestamp, "lower_bound": 100.2, "upper_bound": 100.3})
    approached = BaselineLiquidityAnalyzer(LiquidityConfig(pools=PoolConfig(approach_atr=3)))._lifecycle([approaching], candles[:5], 0.1)[-1][0]
    assert expired.lifecycle_state == LiquidityLifecycleState.EXPIRED
    assert approached.lifecycle_state == LiquidityLifecycleState.APPROACHED


@pytest.mark.asyncio
async def test_memory_repository_idempotency_time_travel_and_recovery() -> None:
    candles = series([(100, 101, 99, 100), (100, 101.01, 99, 100), (100, 102, 98, 101), (101, 103, 99, 102), (102, 104, 100, 103)])
    analyzer = BaselineLiquidityAnalyzer()
    first = analyzer.analyze_snapshot(LiquidityContext(tuple(candles[:-1])))
    second = analyzer.analyze_snapshot(LiquidityContext(tuple(candles)))
    repository = InMemoryLiquidityRepository()
    await repository.save(first)
    await repository.save(first)
    await repository.save(second)
    assert await repository.latest("XAU/USD", Timeframe.M15) == second
    assert await repository.at("XAUUSD", Timeframe.M15, first.analysis_timestamp) == first
    assert await repository.at("XAUUSD", Timeframe.H1, first.analysis_timestamp) is None
    assert await repository.checkpoints() == (second,)


class Scalars:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def first(self) -> object | None:
        return self.values[0] if self.values else None

    def all(self) -> list[object]:
        return self.values


class Session:
    def __init__(self, payload: dict[str, object] | None = None, *, corrupt: bool = False) -> None:
        self.payload = payload
        self.corrupt = corrupt
        self.execute = AsyncMock()
        self.commit = AsyncMock()

    async def scalars(self, statement: object) -> Scalars:
        if self.payload is None:
            return Scalars([])
        if "liquidity_checkpoints" in str(statement):
            return Scalars(
                [
                    SimpleNamespace(
                        state_payload=self.payload if not self.corrupt else {"bad": True},
                        state_hash="bad"
                        if self.corrupt
                        else __import__("hashlib")
                        .sha256(
                            __import__("backend.app.engines.liquidity_engine.models", fromlist=["LiquidityAnalysisSnapshot"])
                            .LiquidityAnalysisSnapshot.model_validate(self.payload)
                            .model_dump_json()
                            .encode()
                        )
                        .hexdigest(),
                        engine_version="1.0.0",
                    )
                ]
            )
        return Scalars([SimpleNamespace(payload=self.payload)])


@pytest.mark.asyncio
async def test_sql_repository_write_query_checkpoint_and_corruption() -> None:
    candles = series([(100, 101, 99, 100), (100, 101.01, 99, 100), (100, 102, 98, 101), (101, 103, 99, 102), (102, 104, 100, 103)])
    snapshot = BaselineLiquidityAnalyzer().analyze_snapshot(LiquidityContext(tuple(candles)))
    session = Session(snapshot.model_dump(mode="json"))
    repository = SqlAlchemyLiquidityRepository(cast(Any, FakeSessionFactory(session)))
    await repository.save(snapshot)
    assert session.execute.await_count == 3 and session.commit.await_count == 1
    assert (await repository.latest("XAUUSD", Timeframe.M15)).id == snapshot.id  # type: ignore[union-attr]
    assert (await repository.at("XAUUSD", Timeframe.M15, snapshot.analysis_timestamp)).id == snapshot.id  # type: ignore[union-attr]
    assert [item.id for item in await repository.checkpoints()] == [snapshot.id]
    assert await SqlAlchemyLiquidityRepository(cast(Any, FakeSessionFactory(Session()))).latest("XAUUSD", Timeframe.M15) is None
    assert await SqlAlchemyLiquidityRepository(cast(Any, FakeSessionFactory(Session(snapshot.model_dump(mode="json"), corrupt=True)))).checkpoints() == ()


class Market:
    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles
        self.sessions = BaselineLiquidityAnalyzer().sessions

    async def history(self, *_: object, **__: object) -> list[Candle]:
        return self.candles

    async def replay(self, _symbol: str, _timeframe: Timeframe, timestamp: datetime, **_: object) -> list[Candle]:
        return [item for item in self.candles if item.timestamp <= timestamp]

    async def close(self) -> None:
        return None


class SMC:
    def __init__(self, candles: list[Candle]) -> None:
        self.context = smc_context(candles)

    async def liquidity_context(self, *_: object) -> SMCLiquidityContext:
        return self.context


@pytest.mark.asyncio
async def test_service_features_events_replay_health_restart_and_failures() -> None:
    candles = series([(100, 101, 99, 100), (100, 101.01, 99, 100), (100, 102, 98, 101), (101, 103, 99, 102), (102, 104, 100, 103)])
    repository = InMemoryLiquidityRepository()
    bus = InMemoryEventBus()
    store = InMemoryFeatureStore()
    service = LiquidityService(cast(Any, Market(candles)), cast(Any, SMC(candles)), bus, store, repository=repository)
    assert await service.restore() == 0 and service.health()["status"] == "degraded"
    assert service.metrics.snapshot()["analyses_completed"] == 0
    snapshot = await service.analyze("XAUUSD", Timeframe.M15)
    published_count = len(bus.history())
    await service._publish(snapshot, uuid4())
    assert len(bus.history()) == published_count
    replay = await service.replay("XAUUSD", Timeframe.M15, candles[-1].timestamp)
    assert replay.processing_mode == ProcessingMode.REPLAY and service.metrics.replay_runs == 1
    restarted = LiquidityService(cast(Any, Market(candles)), cast(Any, SMC(candles)), InMemoryEventBus(), InMemoryFeatureStore(), repository=repository)
    assert await restarted.restore() == 1 and restarted.recovery_status == "recovered"
    assert await restarted.state("XAUUSD", Timeframe.M15) is not None
    values = LiquidityService.features(snapshot)
    assert {
        "nearest_buy_side_liquidity",
        "nearest_sell_side_liquidity",
        "target_rankings",
        "liquidity_map",
        "engine_version",
        "configuration_version",
    } <= values.keys()
    durable = LiquidityService(
        cast(Any, Market(candles)),
        cast(Any, SMC(candles)),
        bus,
        store,
        LiquidityConfig(persistence=PersistenceConfig(required_in_production=True)),
        repository,
        "sqlalchemy",
    )
    assert durable.health()["status"] == "healthy"
    deep = LiquidityService(
        cast(Any, Market(candles)),
        cast(Any, SMC(candles)),
        bus,
        store,
        LiquidityConfig(multi_timeframe=MultiTimeframeConfig(maximum_depth=9)),
        InMemoryLiquidityRepository(),
    )
    mtf = await deep.multi_timeframe("XAUUSD", Timeframe.M15)
    assert {"W1", "MN1"} <= mtf.pools_by_timeframe.keys()
    replay_mtf = await deep.multi_timeframe("XAUUSD", Timeframe.M15, candles[-1].timestamp)
    assert replay_mtf.analyzed_through <= candles[-1].timestamp

    class FailingMarket(Market):
        async def history(self, *_: object, **__: object) -> list[Candle]:
            raise RuntimeError("series")

    failed_mtf = LiquidityService(
        cast(Any, FailingMarket(candles)),
        cast(Any, SMC(candles)),
        bus,
        store,
        LiquidityConfig(multi_timeframe=MultiTimeframeConfig(hierarchy=("H1",), maximum_depth=1)),
    )
    assert not (await failed_mtf.multi_timeframe("XAUUSD", Timeframe.M15)).pools_by_timeframe
    failing = SimpleNamespace(save=AsyncMock(side_effect=RuntimeError("db")))
    broken = LiquidityService(cast(Any, Market(candles)), cast(Any, SMC(candles)), bus, store, repository=cast(Any, failing))
    with pytest.raises(RuntimeError, match="db"):
        await broken.analyze("XAUUSD", Timeframe.M15)
    assert broken.metrics.persistence_failures == 1

    class FailingBus(InMemoryEventBus):
        async def publish(self, event: object) -> None:
            raise RuntimeError("event")

    event_failure = LiquidityService(
        cast(Any, Market(candles)), cast(Any, SMC(candles)), cast(Any, FailingBus()), store, repository=InMemoryLiquidityRepository()
    )
    await event_failure.analyze("XAUUSD", Timeframe.M15)
    assert event_failure.metrics.event_publication_failures == 1


@pytest.mark.asyncio
async def test_registration_uses_smc_contract_and_pipeline_features(candles: list[Candle]) -> None:
    factory = EngineFactory()
    register(factory)
    assert factory.definition("liquidity").metadata.dependencies == ("market_data", "smc")
    engine = _build(cast(EngineBuildContext, object()), {})
    smc_snapshot = SimpleNamespace(
        snapshot=SimpleNamespace(
            symbol="XAUUSD",
            timeframe=Timeframe.M15,
            analysis_timestamp=candles[-1].timestamp,
            structure_state=SimpleNamespace(current_direction=SimpleNamespace(value="bullish")),
            swings=(),
            configuration_version="c",
            engine_version="3",
        )
    )
    context = PipelineExecutionContext(correlation_id=uuid4(), candles=candles, events=[], feature_store=InMemoryFeatureStore(), results={"smc": smc_snapshot})
    result = await _execute(engine, context)
    assert result.namespace == "liquidity" and result.output.snapshot is not None and "target_rankings" in result.features
    context.results.clear()
    assert (await _execute(engine, context)).output.snapshot is not None
