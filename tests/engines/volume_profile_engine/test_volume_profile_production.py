from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.smc_engine.liquidity_contract import SMCLiquidityContext, SMCLiquidityLevel
from backend.app.engines.volume_profile_engine import (
    AnalysisStatus,
    AnchoredVolumeProfile,
    BaselineVolumeProfileAnalyzer,
    CompletedVolumeProfile,
    CompositeVolumeProfile,
    DevelopingVolumeProfile,
    FixedRangeVolumeProfile,
    InMemoryVolumeProfileRepository,
    ProcessingMode,
    ProfileAnchor,
    ProfileLifecycleState,
    ProfileLifecycleTransition,
    ProfileType,
    SessionVolumeProfile,
    VolumeProfileBucket,
    VolumeProfileConfig,
    VolumeProfileContext,
    VolumeProfileService,
    VolumeSourceType,
    validate_transition,
)
from backend.app.engines.volume_profile_engine.config import AllocationConfig, MultiTimeframeConfig, PersistenceConfig, PriceGridConfig
from backend.app.engines.volume_profile_engine.models import AnchorType, ValueArea, VolumeProfile, stable_id, utc, _TRANSITIONS
from backend.app.engines.volume_profile_engine.registration import _build, _execute, register
from backend.app.engines.volume_profile_engine.repository import SqlAlchemyVolumeProfileRepository
from tests.conftest import FakeSessionFactory
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import PipelineExecutionContext


def series(count: int = 12, *, start: datetime | None = None, timeframe: Timeframe = Timeframe.M15, zero: bool = False) -> list[Candle]:
    origin = start or datetime(2026, 1, 1, 7, tzinfo=UTC)
    return [
        Candle(
            timestamp=origin + timeframe.duration * i,
            symbol="XAU/USD",
            timeframe=timeframe,
            open=100 + i % 3,
            high=102 + i % 3,
            low=98 + i % 3,
            close=101 + (i % 3),
            volume=0 if zero else 100 + i,
            quality_score=90,
        )
        for i in range(count)
    ]


def context(candles: list[Candle], source: VolumeSourceType = VolumeSourceType.EXCHANGE, **kwargs: object) -> VolumeProfileContext:
    return VolumeProfileContext(tuple(candles), source, "XAUUSD", 0.01, **kwargs)


def test_configuration_domain_invariants_and_lifecycle() -> None:
    config = VolumeProfileConfig()
    assert config.version == VolumeProfileConfig().version and config.bins == 24 and config.high_volume_percentile == 0.75
    with pytest.raises(ValidationError, match="minimum_bins"):
        PriceGridConfig(minimum_bins=10, maximum_bins=5)
    with pytest.raises(ValidationError, match="unsupported price grid"):
        PriceGridConfig(method="bad")
    with pytest.raises(ValidationError, match="unsupported allocation"):
        AllocationConfig(method="bad")
    with pytest.raises(ValidationError, match="positive sum"):
        AllocationConfig(body_weight=0, wick_weight=0)
    with pytest.raises(ValidationError, match="volume-source policy"):
        VolumeProfileConfig(default_volume_source="tick", allowed_volume_sources=("exchange",))
    for previous, allowed in _TRANSITIONS.items():
        validate_transition(previous, previous)
        for current in allowed:
            validate_transition(previous, current)
    with pytest.raises(ValueError, match="impossible"):
        validate_transition(ProfileLifecycleState.ARCHIVED, ProfileLifecycleState.ACTIVE)
    assert utc(datetime.now(UTC)).tzinfo == UTC
    with pytest.raises(ValueError, match="timezone"):
        utc(datetime.now())


def test_model_validation_and_specialized_serialization() -> None:
    analyzer = BaselineVolumeProfileAnalyzer()
    profile = analyzer.analyze_snapshot(context(series())).profiles[0]
    data = profile.model_dump()
    assert FixedRangeVolumeProfile.model_validate(data).profile_type == ProfileType.FIXED_RANGE
    DevelopingVolumeProfile.model_validate({**data, "profile_type": "developing"})
    CompletedVolumeProfile.model_validate(data)
    CompositeVolumeProfile.model_validate({**data, "profile_type": "composite"})
    AnchoredVolumeProfile.model_validate({**data, "profile_type": "anchored"})
    SessionVolumeProfile.model_validate({**data, "profile_type": "session", "session": "london"})
    bucket = profile.buckets[0]
    with pytest.raises(ValidationError, match="bounds"):
        VolumeProfileBucket.model_validate({**bucket.model_dump(), "upper": bucket.lower})
    with pytest.raises(ValidationError, match="conserve"):
        VolumeProfileBucket.model_validate({**bucket.model_dump(), "estimated_buy_volume": bucket.volume + 1})
    with pytest.raises(ValidationError, match="VAL"):
        ValueArea.model_validate({**profile.value_area.model_dump(), "val": profile.value_area.vah + 1})  # type: ignore[union-attr]
    anchor_time = profile.start_timestamp
    with pytest.raises(ValidationError, match="availability"):
        ProfileAnchor(
            id=uuid4(),
            anchor_type=AnchorType.EXPLICIT,
            anchor_timestamp=anchor_time,
            availability_timestamp=anchor_time - timedelta(1),
            source_engine="manual",
            confidence_score=80,
        )
    with pytest.raises(ValidationError, match="impossible"):
        ProfileLifecycleTransition(
            id=uuid4(),
            profile_id=profile.id,
            previous=ProfileLifecycleState.ARCHIVED,
            current=ProfileLifecycleState.ACTIVE,
            available_at=profile.analysis_boundary,
            reason="bad",
        )
    for update, match in (
        ({"start_timestamp": profile.end_timestamp + timedelta(1)}, "range"),
        ({"availability_timestamp": profile.analysis_boundary + timedelta(1)}, "available"),
        ({"bucket_count": profile.bucket_count + 1}, "bucket_count"),
        ({"included_volume": profile.included_volume + 1}, "conserved"),
        ({"total_volume": 0}, "exceed"),
    ):
        with pytest.raises(ValidationError, match=match):
            VolumeProfile.model_validate({**data, **update})


@pytest.mark.parametrize("method", ["tick", "fixed", "rows", "percentage", "atr", "auto"])
def test_all_price_grid_methods_are_aligned_bounded_and_deterministic(method: str) -> None:
    config = VolumeProfileConfig(price_grid=PriceGridConfig(method=method, rows=12, maximum_bins=20, fixed_increment=0.5, tick_size=0.1))
    analyzer = BaselineVolumeProfileAnalyzer(config)
    first = analyzer.analyze_snapshot(context(series()))
    second = analyzer.analyze_snapshot(context(series()), ProcessingMode.REPLAY)
    profile = first.profiles[0]
    assert profile.bucket_count <= 20 and profile.row_size >= 0.1
    assert profile.buckets[-1].upper_inclusive and not profile.buckets[0].upper_inclusive
    assert [x.id for x in profile.buckets] == [x.id for x in second.profiles[0].buckets]


@pytest.mark.parametrize("method", ["close", "typical_price", "uniform_range", "body_wick"])
def test_allocation_methods_conserve_volume_and_directional_estimate(method: str) -> None:
    config = VolumeProfileConfig(allocation=AllocationConfig(method=method, directional_approximation=True), price_grid=PriceGridConfig(rows=16))
    profile = BaselineVolumeProfileAnalyzer(config).analyze_snapshot(context(series())).profiles[0]
    assert sum(x.volume for x in profile.buckets) == pytest.approx(profile.included_volume)
    assert profile.included_volume == pytest.approx(sum(x.volume for x in series()))
    assert all(x.estimated_buy_volume + x.estimated_sell_volume == pytest.approx(x.volume) for x in profile.buckets)
    assert profile.allocation_method.value == method


def test_empty_insufficient_missing_tick_anchor_periods_and_no_lookahead() -> None:
    analyzer = BaselineVolumeProfileAnalyzer()
    empty = analyzer.analyze([])
    assert empty.snapshot and empty.snapshot.status == AnalysisStatus.INSUFFICIENT_HISTORY and empty.observations
    assert analyzer.analyze_snapshot(context(series(1))).status == AnalysisStatus.INSUFFICIENT_HISTORY
    missing = analyzer.analyze_snapshot(context(series(zero=True), VolumeSourceType.MISSING))
    assert missing.status == AnalysisStatus.DEGRADED and not missing.profiles
    candles = series(60, start=datetime(2026, 1, 30, 7, tzinfo=UTC), timeframe=Timeframe.H1)
    at = candles[10].timestamp
    anchor = ProfileAnchor(
        id=stable_id("anchor", at),
        anchor_type=AnchorType.EXPLICIT,
        anchor_timestamp=at,
        availability_timestamp=at,
        anchor_price=candles[10].close,
        source_engine="manual",
        confidence_score=90,
    )
    tick = analyzer.analyze_snapshot(context(candles, VolumeSourceType.TICK, anchors=(anchor,)))
    assert "tick volume" in tick.volume_data_quality.limitations[0]
    assert {"fixed_range", "developing", "session", "daily", "weekly", "monthly", "composite", "anchored"} <= {x.profile_type.value for x in tick.profiles}
    assert any((x.poc and x.poc.tested) or (x.value_area and (x.value_area.vah_tested or x.value_area.val_tested)) for x in tick.completed)
    prefix = analyzer.analyze_snapshot(context(candles[:30]))
    replay = analyzer.analyze_snapshot(context(candles[:30]), ProcessingMode.REPLAY)
    extended = analyzer.analyze_snapshot(context(candles))
    assert prefix.profiles == replay.profiles and prefix.confluences == replay.confluences and prefix.migrations == replay.migrations
    assert prefix.analysis_timestamp < extended.analysis_timestamp
    assert all(x.availability_timestamp <= prefix.analysis_timestamp for x in prefix.profiles)


def test_nodes_shelves_gaps_shapes_migrations_and_cross_engine_confluence() -> None:
    candles = series(36, timeframe=Timeframe.H1)
    smc = SMCLiquidityContext(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        analyzed_through=candles[-1].timestamp,
        structure_direction="neutral",
        levels=(
            SMCLiquidityLevel(
                id="swing",
                symbol="XAUUSD",
                timeframe=Timeframe.H1,
                kind="swing_high",
                scope="external",
                price=101,
                occurred_at=candles[0].timestamp,
                available_at=candles[1].timestamp,
                confidence_score=90,
                quality_score=90,
            ),
        ),
        configuration_version="c",
        engine_version="3",
    )
    snapshot = BaselineVolumeProfileAnalyzer().analyze_snapshot(context(candles, smc=smc, liquidity_source_ids=("pool",)))
    assert any(x.hvns for x in snapshot.profiles)
    assert snapshot.migrations and snapshot.confluences
    assert {"volume_profile", "smc", "liquidity"} <= set(snapshot.confluences[0].source_types)
    analyzer = BaselineVolumeProfileAnalyzer(VolumeProfileConfig(nodes={"shelf_minimum_width": 2, "gap_maximum_ratio": 0.2}))
    profile = analyzer.analyze_snapshot(context(candles)).profiles[0]
    assert profile.shape and profile.shape.alternative
    # Explicit distributions exercise shelf, internal-gap, double/multimodal, and grouping paths.
    base = profile.buckets[0]
    values = [10, 10, 0, 10, 10, 1, 20, 1]
    buckets = tuple(
        base.model_copy(
            update={
                "id": stable_id("custom", i),
                "index": i,
                "lower": float(i),
                "upper": float(i + 1),
                "midpoint": i + 0.5,
                "volume": float(v),
                "estimated_buy_volume": v / 2,
                "estimated_sell_volume": v / 2,
                "upper_inclusive": i == len(values) - 1,
            }
        )
        for i, v in enumerate(values)
    )
    poc = analyzer._poc(buckets, sum(values), "custom")
    area = analyzer._value_area(buckets, poc, sum(values), "custom")
    hvns, lvns = analyzer._nodes(buckets, "custom")
    shelves, gaps = analyzer._shelves_gaps(buckets, poc, area, "custom")
    shape = analyzer._shape(buckets, poc, hvns, "custom")
    assert hvns and lvns and shelves and gaps and shape.features["mode_count"] >= 1
    assert analyzer._groups([]) == [] and analyzer._groups([1, 2, 4]) == [[1, 2], [4]]

    def shaped(values: list[int], poc_index: int, modes: int = 0) -> str:
        custom = tuple(
            base.model_copy(
                update={
                    "id": stable_id("shape", values, i),
                    "index": i,
                    "lower": float(i),
                    "upper": float(i + 1),
                    "midpoint": i + 0.5,
                    "volume": float(v),
                    "estimated_buy_volume": v / 2,
                    "estimated_sell_volume": v / 2,
                    "upper_inclusive": i == len(values) - 1,
                }
            )
            for i, v in enumerate(values)
        )
        custom_poc = analyzer._poc(custom, sum(values), "shape").model_copy(update={"bucket_id": custom[poc_index].id, "price": custom[poc_index].midpoint})
        nodes = tuple(hvns[0] for _ in range(modes)) if hvns else ()
        return analyzer._shape(custom, custom_poc, nodes, "shape").shape_type.value

    assert shaped([1, 2, 3, 2, 1], 2) == "d_shaped"
    assert shaped([1, 1, 1, 5, 10], 4) == "p_shaped"
    assert shaped([10, 5, 1, 1, 1], 0) == "b_shaped"
    assert shaped([1] * 10, 0) == "thin"
    assert shaped([1, 1, 1, 5, 10], 0) == "trend"
    assert shaped([2, 2, 2, 3, 4], 0) == "undefined"
    assert shaped([1, 5, 1, 5, 1], 2, 2) == "double_distribution"
    assert shaped([1, 5, 1, 5, 1], 2, 3) == "multimodal"

    missing_poc = profile.model_copy(update={"poc": None})
    assert analyzer._migrations([missing_poc, profile]) == ()


@pytest.mark.asyncio
async def test_memory_and_sql_repository_checkpoint_restart_and_corruption() -> None:
    snapshot = BaselineVolumeProfileAnalyzer().analyze_snapshot(context(series()))
    memory = InMemoryVolumeProfileRepository()
    await memory.save(snapshot)
    await memory.save(snapshot)
    assert await memory.latest("XAU/USD", Timeframe.M15) == snapshot
    assert await memory.at("XAUUSD", Timeframe.M15, snapshot.analysis_timestamp) == snapshot
    assert await memory.at("XAUUSD", Timeframe.H1, snapshot.analysis_timestamp) is None
    assert await memory.checkpoints() == (snapshot,)

    class Scalars:
        def __init__(self, values: list[object]):
            self.values = values

        def first(self) -> object | None:
            return self.values[0] if self.values else None

        def all(self) -> list[object]:
            return self.values

    class Session:
        def __init__(self, payload: dict[str, object] | None = None, corrupt: bool = False):
            self.payload, self.corrupt = payload, corrupt
            self.execute = AsyncMock()
            self.commit = AsyncMock()

        async def scalars(self, statement: object) -> Scalars:
            if self.payload is None:
                return Scalars([])
            if "volume_profile_checkpoints" in str(statement):
                import hashlib

                return Scalars(
                    [
                        SimpleNamespace(
                            state_payload={"bad": True} if self.corrupt else self.payload,
                            state_hash="bad"
                            if self.corrupt
                            else hashlib.sha256(BaselineVolumeProfileAnalyzer().analyze_snapshot(context(series())).model_dump_json().encode()).hexdigest(),
                            engine_version="1.0.0",
                        )
                    ]
                )
            return Scalars([SimpleNamespace(payload=self.payload)])

    session = Session(snapshot.model_dump(mode="json"))
    sql = SqlAlchemyVolumeProfileRepository(cast(Any, FakeSessionFactory(session)))
    await sql.save(snapshot)
    assert session.execute.await_count == 3 and session.commit.await_count == 1
    assert (await sql.latest("XAUUSD", Timeframe.M15)).id == snapshot.id  # type: ignore[union-attr]
    assert (await sql.at("XAUUSD", Timeframe.M15, snapshot.analysis_timestamp)).id == snapshot.id  # type: ignore[union-attr]
    assert (await sql.checkpoints())[0].id == snapshot.id
    assert await SqlAlchemyVolumeProfileRepository(cast(Any, FakeSessionFactory(Session()))).latest("XAUUSD", Timeframe.M15) is None
    assert await SqlAlchemyVolumeProfileRepository(cast(Any, FakeSessionFactory(Session(snapshot.model_dump(mode="json"), True)))).checkpoints() == ()


class Market:
    def __init__(self, candles: list[Candle]):
        self.candles = candles
        self.sessions = BaselineVolumeProfileAnalyzer().sessions

    async def history(self, *_: object, **__: object) -> list[Candle]:
        return self.candles

    async def replay(self, _s: str, _t: Timeframe, at: datetime, **_: object) -> list[Candle]:
        return [x for x in self.candles if x.timestamp <= at]


class SMC:
    async def liquidity_context(self, symbol: str, timeframe: Timeframe, at: datetime) -> SMCLiquidityContext:
        return SMCLiquidityContext(
            symbol=symbol, timeframe=timeframe, analyzed_through=at, structure_direction="neutral", configuration_version="c", engine_version="3"
        )


class Liquidity:
    async def state(self, *_: object) -> object:
        return SimpleNamespace(pools=(SimpleNamespace(id=uuid4()),))


@pytest.mark.asyncio
async def test_service_features_events_replay_mtf_health_failures_and_registration() -> None:
    candles, repository, bus, store = series(), InMemoryVolumeProfileRepository(), InMemoryEventBus(), InMemoryFeatureStore()
    service = VolumeProfileService(
        cast(Any, Market(candles)), SMC(), Liquidity(), bus, store, VolumeProfileConfig(default_volume_source="exchange"), repository
    )
    assert await service.restore() == 0 and service.health()["status"] == "degraded"
    snapshot = await service.analyze("XAUUSD", Timeframe.M15)
    count = len(bus.history())
    await service._publish(snapshot, uuid4())
    assert len(bus.history()) == count
    replay = await service.replay("XAUUSD", Timeframe.M15, candles[-1].timestamp)
    assert replay.processing_mode == ProcessingMode.REPLAY and service.metrics.replay_runs == 1
    restarted = VolumeProfileService(cast(Any, Market(candles)), SMC(), Liquidity(), InMemoryEventBus(), InMemoryFeatureStore(), VolumeProfileConfig(default_volume_source="exchange"), repository)
    assert await restarted.restore() == 1 and restarted.recovery_status == "recovered"
    assert await restarted.state("XAUUSD", Timeframe.M15) is not None
    assert {"developing_poc", "active_shelves", "previous_day", "source_traceability", "configuration_version"} <= service.features(snapshot).keys()
    mismatch = VolumeProfileService(
        cast(Any, Market(candles)),
        SMC(),
        Liquidity(),
        InMemoryEventBus(),
        InMemoryFeatureStore(),
        VolumeProfileConfig(price_grid=PriceGridConfig(rows=25)),
        repository,
    )
    assert await mismatch.restore() == 0
    assert "volume_source_semantics_low_confidence" in mismatch.health()["degraded_reasons"]
    durable = VolumeProfileService(
        cast(Any, Market(candles)),
        SMC(),
        Liquidity(),
        bus,
        store,
        VolumeProfileConfig(default_volume_source="exchange", persistence=PersistenceConfig(required_in_production=True)),
        repository,
        "sqlalchemy",
    )
    durable.metrics.latest_successful_analysis_timestamp = "now"
    assert durable.health()["status"] == "healthy"
    deep = VolumeProfileService(
        cast(Any, Market(candles)),
        SMC(),
        Liquidity(),
        bus,
        store,
        VolumeProfileConfig(default_volume_source="exchange", multi_timeframe=MultiTimeframeConfig(maximum_depth=9)),
        InMemoryVolumeProfileRepository(),
    )
    assert {"W1", "MN1"} <= (await deep.multi_timeframe("XAUUSD", Timeframe.M15)).profile_ids_by_timeframe.keys()
    assert (await deep.multi_timeframe("XAUUSD", Timeframe.M15, candles[-1].timestamp)).analyzed_through <= candles[-1].timestamp

    class FailingMarket(Market):
        async def history(self, *_: object, **__: object) -> list[Candle]:
            raise RuntimeError("market")

    failed_mtf = VolumeProfileService(
        cast(Any, FailingMarket(candles)),
        SMC(),
        Liquidity(),
        bus,
        store,
        VolumeProfileConfig(multi_timeframe=MultiTimeframeConfig(hierarchy=("H1",), maximum_depth=1)),
    )
    assert not (await failed_mtf.multi_timeframe("XAUUSD", Timeframe.M15)).profile_ids_by_timeframe

    failing = SimpleNamespace(save=AsyncMock(side_effect=RuntimeError("db")))
    broken = VolumeProfileService(cast(Any, Market(candles)), SMC(), Liquidity(), bus, store, repository=cast(Any, failing))
    with pytest.raises(RuntimeError, match="db"):
        await broken.analyze("XAUUSD", Timeframe.M15)
    assert broken.metrics.persistence_failures == 1

    class FailingBus(InMemoryEventBus):
        async def publish(self, event: object) -> None:
            raise RuntimeError("event")

    failed_events = VolumeProfileService(
        cast(Any, Market(candles)),
        SMC(),
        Liquidity(),
        cast(Any, FailingBus()),
        store,
        VolumeProfileConfig(default_volume_source="exchange"),
        InMemoryVolumeProfileRepository(),
    )
    await failed_events.analyze("XAUUSD", Timeframe.M15)
    assert failed_events.metrics.event_publication_failures == 1

    class FailingStore(InMemoryFeatureStore):
        async def write(self, feature: object) -> None:
            raise RuntimeError("feature")

    failed_feature = VolumeProfileService(
        cast(Any, Market(candles)),
        SMC(),
        Liquidity(),
        bus,
        cast(Any, FailingStore()),
        VolumeProfileConfig(default_volume_source="exchange"),
        InMemoryVolumeProfileRepository(),
    )
    await failed_feature.analyze("XAUUSD", Timeframe.M15)
    assert failed_feature.metrics.feature_publication_failures == 1

    factory = EngineFactory()
    register(factory)
    assert factory.definition("volume_profile").metadata.dependencies == ("market_data", "smc", "liquidity")
    engine = _build(cast(EngineBuildContext, object()), {})
    result = await _execute(engine, PipelineExecutionContext(correlation_id=uuid4(), candles=candles, events=[], feature_store=store, results={}))
    assert result.namespace == "volume_profile" and result.features["poc"] is not None
    smc_result = SimpleNamespace(snapshot=SimpleNamespace(liquidity_context=object()))
    assert (
        await _execute(engine, PipelineExecutionContext(correlation_id=uuid4(), candles=candles, events=[], feature_store=store, results={"smc": smc_result}))
    ).output.poc is not None

    anchor = ProfileAnchor(
        id=uuid4(),
        anchor_type=AnchorType.EXPLICIT,
        anchor_timestamp=candles[1].timestamp,
        availability_timestamp=candles[1].timestamp,
        source_engine="manual",
        confidence_score=90,
    )
    anchored = await service.analyze_context(context(candles, anchors=(anchor,)))
    assert any(x.profile_type == ProfileType.ANCHORED for x in anchored.profiles)
