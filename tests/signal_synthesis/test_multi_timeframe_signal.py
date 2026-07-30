from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest

from backend.app.ai_reasoning.analysis import (
    AIMarketAnalysis,
    AnalysisBias,
    AnalysisEvidence,
    EvidenceKind,
)
from backend.app.market_state import EvidenceAvailability
from backend.app.signal_synthesis import (
    AnalyticalDirection,
    AnalyticalStrength,
    ExecutionEligibility,
    InMemoryMultiTimeframeSignalRepository,
    MultiTimeframeSignalSynthesizer,
)
from tests.ai_reasoning.test_ai_reasoning_lifecycle import state_and_quant
from tests.ai_reasoning.test_analysis_architecture_v2 import analysis


def aligned_analysis(state, quant) -> AIMarketAnalysis:
    value = analysis(0)
    return value.model_copy(
        update={
            "cycle_id": state.cycle_id,
            "market_snapshot_id": state.state_id,
            "quantitative_forecast_id": quant.result_id,
            "analysis_timestamp": state.market_data_boundary,
            "knowledge_cutoff": state.knowledge_cutoff,
            "created_at": state.knowledge_cutoff,
        }
    )


def with_smc_zones(state, timeframe: str, zones: list[dict[str, object]]):
    evidence = []
    for item in state.evidence:
        if item.source_timeframe == timeframe and item.source_engine == "smc":
            raw = deepcopy(item.raw_value)
            raw["zones"] = zones
            item = item.model_copy(update={"raw_value": raw})
        evidence.append(item)
    return state.model_copy(update={"evidence": tuple(evidence)})


def with_raw_evidence(state, timeframe: str, engine: str, raw_value: dict[str, object]):
    evidence = tuple(
        item.model_copy(update={"raw_value": raw_value})
        if item.source_timeframe == timeframe and item.source_engine == engine
        else item
        for item in state.evidence
    )
    return state.model_copy(update={"evidence": evidence})


def with_market(
    state,
    timeframe: str,
    *,
    open_price: float,
    high: float,
    low: float,
    close: float,
    spread: float = 0.2,
):
    market = next(
        item
        for item in state.evidence
        if item.source_timeframe == timeframe
        and item.source_engine == "market_data"
    )
    raw = deepcopy(market.raw_value)
    raw.update(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "spread": spread,
        }
    )
    return with_raw_evidence(state, timeframe, "market_data", raw)


def bearish_order_block(identifier: str, lifecycle: str = "active") -> dict[str, object]:
    return {
        "id": identifier,
        "zone_type": "bearish_order_block",
        "direction": "bearish",
        "lifecycle_state": lifecycle,
        "lower_price": 3303,
        "upper_price": 3306,
        "midpoint": 3304.5,
        "mitigation_percentage": 0,
        "quality_score": 100,
        "source_candle_ids": [f"candle-{identifier}"],
    }


@pytest.mark.asyncio
async def test_completed_cycle_produces_independent_m5_m15_and_combined_directions() -> None:
    state, quant = await state_and_quant()
    result = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, aligned_analysis(state, quant)
    )

    assert tuple(item.timeframe for item in result.timeframe_signals) == ("M5", "M15")
    assert all(item.analytical_direction in {AnalyticalDirection.BUY, AnalyticalDirection.SELL} for item in result.timeframe_signals)
    assert result.combined_signal.analytical_direction in {AnalyticalDirection.BUY, AnalyticalDirection.SELL}
    assert "HOLD" not in result.model_dump_json()
    assert "WAIT" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_fresh_order_blocks_materially_change_only_their_own_timeframe() -> None:
    state, quant = await state_and_quant()
    baseline = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, aligned_analysis(state, quant)
    )
    changed_state = with_smc_zones(
        state,
        "M5",
        [bearish_order_block(f"ob-{index}") for index in range(3)],
    )
    changed = MultiTimeframeSignalSynthesizer().synthesize(
        changed_state, quant, aligned_analysis(changed_state, quant)
    )

    before = {item.timeframe: item for item in baseline.timeframe_signals}
    after = {item.timeframe: item for item in changed.timeframe_signals}
    assert before["M15"].bullish_score == after["M15"].bullish_score
    assert after["M5"].bearish_score > before["M5"].bearish_score
    assert any(item.family == "order_block" for item in after["M5"].evidence_breakdown)


@pytest.mark.asyncio
async def test_mitigated_order_block_is_not_fresh_confirmation() -> None:
    state, quant = await state_and_quant()
    changed_state = with_smc_zones(
        state,
        "M5",
        [bearish_order_block("mitigated", lifecycle="mitigated")],
    )
    result = MultiTimeframeSignalSynthesizer().synthesize(
        changed_state, quant, aligned_analysis(changed_state, quant)
    )
    m5 = next(item for item in result.timeframe_signals if item.timeframe == "M5")
    assert all("mitigated" not in item.source_fact_identifiers for item in m5.evidence_breakdown)


@pytest.mark.asyncio
async def test_correlated_order_blocks_are_discounted_and_not_counted_as_independent() -> None:
    state, quant = await state_and_quant()
    changed_state = with_smc_zones(
        state,
        "M5",
        [bearish_order_block(f"correlated-{index}") for index in range(3)],
    )
    result = MultiTimeframeSignalSynthesizer().synthesize(
        changed_state, quant, aligned_analysis(changed_state, quant)
    )
    order_blocks = [
        item
        for item in next(signal for signal in result.timeframe_signals if signal.timeframe == "M5").evidence_breakdown
        if item.family == "order_block"
    ]
    assert [item.correlated_discount for item in order_blocks] == [1.0, 0.35, 0.35]


@pytest.mark.asyncio
async def test_volume_profile_migration_uses_poc_change_not_a_nonexistent_direction_field() -> None:
    state, quant = await state_and_quant()
    changed_state = with_raw_evidence(
        state,
        "M5",
        "volume_profile",
        {
            "id": "vp-m5",
            "migrations": [
                {
                    "id": "migration-up",
                    "migration_type": "upward",
                    "poc_change": 1.25,
                    "normalized_change": 0.8,
                    "quality_score": 90,
                }
            ],
        },
    )
    result = MultiTimeframeSignalSynthesizer().synthesize(
        changed_state, quant, aligned_analysis(changed_state, quant)
    )
    m5 = next(item for item in result.timeframe_signals if item.timeframe == "M5")
    migration = next(
        item
        for item in m5.evidence_breakdown
        if item.family == "volume"
    )

    assert migration.directional_contribution == AnalyticalDirection.BUY
    assert migration.source_fact_identifiers == ("migration-up", "vp-m5")
    assert migration.normalized_score == 0.8


@pytest.mark.asyncio
async def test_unavailable_engine_cannot_contribute_direction_and_blocks_execution() -> None:
    state, quant = await state_and_quant()
    evidence = tuple(
        item.model_copy(
            update={
                "availability": EvidenceAvailability.UNAVAILABLE,
                "reason_codes": ("provider_unavailable",),
            }
        )
        if item.source_timeframe == "M5" and item.source_engine == "institutional_flow"
        else item
        for item in state.evidence
    )
    changed_state = state.model_copy(update={"evidence": evidence})
    result = MultiTimeframeSignalSynthesizer().synthesize(
        changed_state, quant, aligned_analysis(changed_state, quant)
    )
    m5 = next(item for item in result.timeframe_signals if item.timeframe == "M5")

    assert all(item.tool != "institutional_flow" for item in m5.evidence_breakdown)
    assert "timeframe_evidence_degraded" in m5.blocking_reasons
    assert m5.execution_eligibility == ExecutionEligibility.INELIGIBLE


@pytest.mark.asyncio
async def test_liquidity_sweep_reclaim_and_continuation_have_opposite_contributions() -> None:
    state, quant = await state_and_quant()
    reclaimed_state = with_raw_evidence(
        state,
        "M5",
        "liquidity",
        {
            "sweeps": [
                {
                    "id": "sell-side-reclaim",
                    "pool_id": "sell-side-pool",
                    "side": "sell_side",
                    "classification": "wick_only",
                    "reclaim_timestamp": "2026-07-23T12:29:00Z",
                    "reclaim_strength": 80,
                    "quality_score": 90,
                }
            ]
        },
    )
    continuation_state = with_raw_evidence(
        state,
        "M5",
        "liquidity",
        {
            "sweeps": [
                {
                    "id": "sell-side-continuation",
                    "pool_id": "sell-side-pool",
                    "side": "sell_side",
                    "classification": "continuation",
                    "reclaim_timestamp": None,
                    "reclaim_strength": 0,
                    "quality_score": 90,
                }
            ]
        },
    )
    reclaimed = MultiTimeframeSignalSynthesizer().synthesize(
        reclaimed_state, quant, aligned_analysis(reclaimed_state, quant)
    )
    continued = MultiTimeframeSignalSynthesizer().synthesize(
        continuation_state, quant, aligned_analysis(continuation_state, quant)
    )
    reclaimed_fact = next(
        item
        for signal in reclaimed.timeframe_signals
        if signal.timeframe == "M5"
        for item in signal.evidence_breakdown
        if item.tool == "liquidity"
    )
    continued_fact = next(
        item
        for signal in continued.timeframe_signals
        if signal.timeframe == "M5"
        for item in signal.evidence_breakdown
        if item.tool == "liquidity"
    )

    assert reclaimed_fact.directional_contribution == AnalyticalDirection.BUY
    assert continued_fact.directional_contribution == AnalyticalDirection.SELL


@pytest.mark.asyncio
async def test_independent_confluence_raises_confidence_over_weak_evidence() -> None:
    state, quant = await state_and_quant()
    weak_state = state.model_copy(
        update={
            "evidence": tuple(
                item for item in state.evidence if item.source_engine == "market_data"
            )
        }
    )
    weak = MultiTimeframeSignalSynthesizer().synthesize(
        weak_state, quant, aligned_analysis(weak_state, quant)
    )
    confluence_state = with_raw_evidence(
        with_raw_evidence(
            with_raw_evidence(
                state,
                "M5",
                "market_regime",
                {"net_directional_score": 0.8, "compression_score": 0},
            ),
            "M5",
            "institutional_flow",
            {
                "id": "flow-m5",
                "state": {
                    "pressure": {
                        "net_pressure": 0.8,
                        "quality": 0.9,
                    }
                },
            },
        ),
        "M5",
        "volume_profile",
        {
            "id": "vp-m5",
            "migrations": [
                {
                    "id": "migration-up",
                    "migration_type": "upward",
                    "poc_change": 2,
                    "normalized_change": 0.9,
                    "quality_score": 90,
                }
            ],
        },
    )
    confluence = MultiTimeframeSignalSynthesizer().synthesize(
        confluence_state, quant, aligned_analysis(confluence_state, quant)
    )
    weak_m5 = next(item for item in weak.timeframe_signals if item.timeframe == "M5")
    confluence_m5 = next(
        item for item in confluence.timeframe_signals if item.timeframe == "M5"
    )

    assert weak_m5.analytical_direction == AnalyticalDirection.BUY
    assert weak_m5.strength in {
        AnalyticalStrength.VERY_WEAK,
        AnalyticalStrength.WEAK,
        AnalyticalStrength.MODERATE,
    }
    assert confluence_m5.confidence > weak_m5.confidence
    assert (
        confluence_m5.confidence_decomposition.independent_confluence
        > weak_m5.confidence_decomposition.independent_confluence
    )


@pytest.mark.asyncio
async def test_ai_cannot_introduce_an_unsupported_market_fact() -> None:
    state, quant = await state_and_quant()
    current_analysis = aligned_analysis(state, quant)
    assert current_analysis.output is not None
    unsupported = AnalysisEvidence(
        claim="A provider invented an order block.",
        kind=EvidenceKind.AI_INTERPRETATION,
        source_type="smc",
        source_reference="not-a-persisted-evidence-id",
        timeframe="M5",
        observed_value="bullish_order_block",
    )
    changed_output = current_analysis.output.model_copy(
        update={"bullish_evidence": (unsupported,), "bearish_evidence": ()}
    )
    changed_analysis = current_analysis.model_copy(update={"output": changed_output})
    result = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, changed_analysis
    )

    assert all(
        item.family != "ai_interpretation"
        for signal in result.timeframe_signals
        for item in signal.evidence_breakdown
    )


@pytest.mark.asyncio
async def test_quant_ai_disagreement_is_preserved_and_reduces_confidence() -> None:
    state, quant = await state_and_quant()
    agreeing_analysis = aligned_analysis(state, quant)
    assert agreeing_analysis.output is not None
    agreeing = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, agreeing_analysis
    )
    disagreeing_momentum = agreeing_analysis.output.momentum_analysis.model_copy(
        update={"direction": AnalysisBias.BEARISH}
    )
    disagreeing_output = agreeing_analysis.output.model_copy(
        update={"momentum_analysis": disagreeing_momentum}
    )
    disagreeing_analysis = agreeing_analysis.model_copy(
        update={"output": disagreeing_output}
    )
    disagreeing = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, disagreeing_analysis
    )
    agreeing_m5 = next(
        item for item in agreeing.timeframe_signals if item.timeframe == "M5"
    )
    disagreeing_m5 = next(
        item for item in disagreeing.timeframe_signals if item.timeframe == "M5"
    )

    assert agreeing_m5.analytical_direction == disagreeing_m5.analytical_direction
    assert agreeing_m5.confidence_decomposition.quant_ai_alignment == 100
    assert disagreeing_m5.confidence_decomposition.quant_ai_alignment == 25
    assert disagreeing_m5.confidence < agreeing_m5.confidence


@pytest.mark.asyncio
async def test_geometry_uses_only_active_zone_and_liquidity_fact_identifiers() -> None:
    state, quant = await state_and_quant()
    geometry_state = with_smc_zones(
        state,
        "M5",
        [
            {
                "id": "demand-entry",
                "zone_type": "bullish_order_block",
                "direction": "bullish",
                "lifecycle_state": "active",
                "lower_price": 3298,
                "upper_price": 3300,
                "midpoint": 3299,
                "mitigation_percentage": 0,
                "quality_score": 100,
                "source_candle_ids": ["entry-source"],
                "origin_timestamp": (
                    state.market_data_boundary - timedelta(hours=6)
                ).isoformat(),
            }
        ],
    )
    geometry_state = with_raw_evidence(
        geometry_state,
        "M5",
        "liquidity",
        {
            "pools": [
                {
                    "id": "target-pool",
                    "side": "buy_side",
                    "lifecycle_state": "active",
                    "lower_bound": 3308,
                    "upper_bound": 3308,
                }
            ],
            "targets": [
                {
                    "id": "target-fact",
                    "pool_id": "target-pool",
                    "side": "buy_side",
                    "status": "active",
                    "accessibility_score": 90,
                    "confidence_score": 90,
                }
            ],
        },
    )
    result = MultiTimeframeSignalSynthesizer().synthesize(
        geometry_state, quant, aligned_analysis(geometry_state, quant)
    )
    m5 = next(item for item in result.timeframe_signals if item.timeframe == "M5")

    assert m5.analytical_direction == AnalyticalDirection.BUY
    assert m5.geometry is not None
    assert m5.geometry.entry == 3300
    assert m5.geometry.stop_loss == pytest.approx(3297.65)
    assert m5.geometry.take_profit == 3308
    assert m5.geometry.risk_reward_ratio == pytest.approx(3.40425532)
    assert m5.geometry.validated_market_price == 3302
    assert m5.geometry.source_timeframe == "M5"
    assert m5.geometry.maximum_entry_distance == pytest.approx(9.906)
    assert "demand-entry" in m5.geometry.basis_fact_identifiers
    assert "target-fact" in m5.geometry.basis_fact_identifiers


@pytest.mark.asyncio
async def test_buy_geometry_is_rejected_after_market_has_travelled_beyond_target() -> None:
    state, _ = await state_and_quant()
    changed = with_market(
        state,
        "M5",
        open_price=4048,
        high=4051,
        low=4047,
        close=4049,
    )
    changed = with_smc_zones(
        changed,
        "M5",
        [
            {
                "id": "historical-demand",
                "zone_type": "bullish_order_block",
                "lifecycle_state": "active",
                "lower_price": 4024,
                "upper_price": 4026,
            }
        ],
    )
    changed = with_raw_evidence(
        changed,
        "M5",
        "liquidity",
        {
            "pools": [
                {
                    "id": "already-reached-target",
                    "side": "buy_side",
                    "lifecycle_state": "active",
                    "lower_bound": 4041,
                    "upper_bound": 4041,
                }
            ],
            "targets": [
                {
                    "id": "already-reached-target-fact",
                    "pool_id": "already-reached-target",
                    "side": "buy_side",
                    "status": "active",
                }
            ],
        },
    )
    engine = MultiTimeframeSignalSynthesizer()
    evidence = tuple(
        item for item in changed.evidence if item.source_timeframe == "M5"
    )

    geometry, reasons, _ = engine._geometry(
        AnalyticalDirection.BUY,
        "M5",
        evidence,
        engine._market_context(changed),
        evaluated_at=changed.market_data_boundary,
        valid_until=changed.market_data_boundary + timedelta(minutes=15),
    )

    assert geometry is None
    assert reasons == ("no_reachable_directionally_aligned_entry_zone",)


@pytest.mark.asyncio
async def test_sell_geometry_is_rejected_when_target_is_above_current_market() -> None:
    state, _ = await state_and_quant()
    changed = with_market(
        state,
        "M5",
        open_price=3289,
        high=3290,
        low=3287,
        close=3288,
    )
    changed = with_smc_zones(
        changed,
        "M5",
        [
            {
                "id": "old-supply",
                "zone_type": "bearish_order_block",
                "lifecycle_state": "active",
                "lower_price": 3303,
                "upper_price": 3306,
            }
        ],
    )
    changed = with_raw_evidence(
        changed,
        "M5",
        "liquidity",
        {
            "pools": [
                {
                    "id": "sell-target",
                    "side": "sell_side",
                    "lifecycle_state": "active",
                    "lower_bound": 3295,
                    "upper_bound": 3295,
                }
            ]
        },
    )
    engine = MultiTimeframeSignalSynthesizer()
    evidence = tuple(
        item for item in changed.evidence if item.source_timeframe == "M5"
    )

    geometry, reasons, _ = engine._geometry(
        AnalyticalDirection.SELL,
        "M5",
        evidence,
        engine._market_context(changed),
        evaluated_at=changed.market_data_boundary,
        valid_until=changed.market_data_boundary + timedelta(minutes=15),
    )

    assert geometry is None
    assert reasons == ("no_reachable_directionally_aligned_entry_zone",)


@pytest.mark.asyncio
async def test_valid_sell_geometry_keeps_target_below_live_market_and_entry_nearby() -> None:
    state, _ = await state_and_quant()
    changed = with_smc_zones(
        state,
        "M5",
        [
            {
                "id": "fresh-supply",
                "zone_type": "bearish_order_block",
                "lifecycle_state": "active",
                "lower_price": 3303,
                "upper_price": 3306,
            }
        ],
    )
    changed = with_raw_evidence(
        changed,
        "M5",
        "liquidity",
        {
            "pools": [
                {
                    "id": "sell-side-pool",
                    "side": "sell_side",
                    "lifecycle_state": "active",
                    "lower_bound": 3290,
                    "upper_bound": 3290,
                }
            ]
        },
    )
    engine = MultiTimeframeSignalSynthesizer()
    evidence = tuple(
        item for item in changed.evidence if item.source_timeframe == "M5"
    )
    context = engine._market_context(changed)

    geometry, reasons, _ = engine._geometry(
        AnalyticalDirection.SELL,
        "M5",
        evidence,
        context,
        evaluated_at=changed.market_data_boundary,
        valid_until=changed.market_data_boundary + timedelta(minutes=15),
    )

    assert reasons == ()
    assert context is not None
    assert geometry is not None
    assert geometry.take_profit < context.price
    assert abs(geometry.entry - context.price) <= geometry.maximum_entry_distance
    assert geometry.take_profit < geometry.entry < geometry.stop_loss
    assert geometry.risk_reward_ratio >= engine.config.minimum_risk_reward


@pytest.mark.asyncio
async def test_current_candle_that_already_crossed_buy_target_rejects_geometry() -> None:
    state, _ = await state_and_quant()
    changed = with_smc_zones(
        state,
        "M5",
        [
            {
                "id": "fresh-demand",
                "zone_type": "bullish_order_block",
                "lifecycle_state": "active",
                "lower_price": 3298,
                "upper_price": 3300,
            }
        ],
    )
    changed = with_raw_evidence(
        changed,
        "M5",
        "liquidity",
        {
            "pools": [
                {
                    "id": "crossed-buy-target",
                    "side": "buy_side",
                    "lifecycle_state": "active",
                    "lower_bound": 3304,
                    "upper_bound": 3304,
                }
            ]
        },
    )
    engine = MultiTimeframeSignalSynthesizer()
    evidence = tuple(
        item for item in changed.evidence if item.source_timeframe == "M5"
    )

    geometry, reasons, _ = engine._geometry(
        AnalyticalDirection.BUY,
        "M5",
        evidence,
        engine._market_context(changed),
        evaluated_at=changed.market_data_boundary,
        valid_until=changed.market_data_boundary + timedelta(minutes=15),
    )

    assert geometry is None
    assert reasons == ("buy_target_already_traversed",)


@pytest.mark.asyncio
async def test_expired_historical_structure_never_becomes_active_geometry() -> None:
    state, _ = await state_and_quant()
    changed = with_smc_zones(
        state,
        "M5",
        [
            {
                "id": "expired-demand",
                "zone_type": "bullish_order_block",
                "lifecycle_state": "active",
                "lower_price": 3298,
                "upper_price": 3300,
                "expiration_timestamp": (
                    state.market_data_boundary - timedelta(seconds=1)
                ).isoformat(),
            }
        ],
    )
    engine = MultiTimeframeSignalSynthesizer()
    evidence = tuple(
        item for item in changed.evidence if item.source_timeframe == "M5"
    )

    geometry, reasons, _ = engine._geometry(
        AnalyticalDirection.BUY,
        "M5",
        evidence,
        engine._market_context(changed),
        evaluated_at=changed.market_data_boundary,
        valid_until=changed.market_data_boundary + timedelta(minutes=15),
    )

    assert geometry is None
    assert reasons == ("structural_setup_expired",)


@pytest.mark.asyncio
async def test_insufficient_remaining_validity_never_produces_geometry() -> None:
    state, _ = await state_and_quant()
    engine = MultiTimeframeSignalSynthesizer()
    evidence = tuple(
        item for item in state.evidence if item.source_timeframe == "M5"
    )

    geometry, reasons, _ = engine._geometry(
        AnalyticalDirection.BUY,
        "M5",
        evidence,
        engine._market_context(state),
        evaluated_at=state.market_data_boundary,
        valid_until=state.market_data_boundary + timedelta(seconds=30),
    )

    assert geometry is None
    assert reasons == ("insufficient_remaining_validity",)


@pytest.mark.asyncio
async def test_m15_geometry_uses_freshest_synchronized_m5_market_context() -> None:
    state, _ = await state_and_quant()
    changed = with_market(
        state,
        "M5",
        open_price=4048,
        high=4051,
        low=4047,
        close=4049,
    )
    changed = with_smc_zones(
        changed,
        "M15",
        [
            {
                "id": "m15-old-demand",
                "zone_type": "bullish_order_block",
                "lifecycle_state": "active",
                "lower_price": 3298,
                "upper_price": 3300,
            }
        ],
    )
    engine = MultiTimeframeSignalSynthesizer()
    evidence = tuple(
        item for item in changed.evidence if item.source_timeframe == "M15"
    )
    context = engine._market_context(changed)

    assert context is not None
    assert context.price == 4049
    geometry, reasons, _ = engine._geometry(
        AnalyticalDirection.BUY,
        "M15",
        evidence,
        context,
        evaluated_at=changed.market_data_boundary,
        valid_until=changed.market_data_boundary + timedelta(minutes=15),
    )
    assert geometry is None
    assert reasons == ("no_reachable_directionally_aligned_entry_zone",)


@pytest.mark.asyncio
async def test_combined_signal_preserves_opposing_timeframes_and_structural_weights() -> None:
    state, quant = await state_and_quant()
    changed = state
    for timeframe in ("M15",):
        market = next(
            item
            for item in changed.evidence
            if item.source_timeframe == timeframe
            and item.source_engine == "market_data"
        )
        market_raw = deepcopy(market.raw_value)
        market_raw.update({"open": 3303, "high": 3304, "low": 3298, "close": 3299})
        changed = with_raw_evidence(
            changed,
            timeframe,
            "market_data",
            market_raw,
        )
        changed = with_raw_evidence(
            changed,
            timeframe,
            "market_regime",
            {"net_directional_score": -1.0, "compression_score": 0},
        )
        changed = with_raw_evidence(
            changed,
            timeframe,
            "institutional_flow",
            {
                "id": f"flow-{timeframe}",
                "state": {
                    "pressure": {
                        "net_pressure": -1.0,
                        "quality": 1.0,
                    }
                },
            },
        )
        changed = with_raw_evidence(
            changed,
            timeframe,
            "volume_profile",
            {
                "id": f"vp-{timeframe}",
                "migrations": [
                    {
                        "id": f"migration-{timeframe}",
                        "migration_type": "downward",
                        "poc_change": -2,
                        "normalized_change": 1,
                        "quality_score": 100,
                    }
                ],
            },
        )
    result = MultiTimeframeSignalSynthesizer().synthesize(
        changed, quant, aligned_analysis(changed, quant)
    )
    directions = {
        item.timeframe: item.analytical_direction
        for item in result.timeframe_signals
    }

    assert directions == {
        "M5": AnalyticalDirection.BUY,
        "M15": AnalyticalDirection.SELL,
    }
    assert result.combined_signal.analytical_direction == AnalyticalDirection.SELL
    assert tuple(item.timeframe for item in result.timeframe_contributions) == (
        "M5",
        "M15",
    )
    assert "outweighs the opposing lower-timeframe scenario" in (
        result.combined_signal.directional_thesis
    )


@pytest.mark.asyncio
async def test_regime_compression_reduces_confidence_without_erasing_direction() -> None:
    state, quant = await state_and_quant()
    baseline = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, aligned_analysis(state, quant)
    )
    regime = next(
        item
        for item in state.evidence
        if item.source_timeframe == "M5" and item.source_engine == "market_regime"
    )
    raw = deepcopy(regime.raw_value)
    raw["compression_score"] = 1.0
    compressed_state = with_raw_evidence(state, "M5", "market_regime", raw)
    compressed = MultiTimeframeSignalSynthesizer().synthesize(
        compressed_state, quant, aligned_analysis(compressed_state, quant)
    )
    before = next(item for item in baseline.timeframe_signals if item.timeframe == "M5")
    after = next(item for item in compressed.timeframe_signals if item.timeframe == "M5")

    assert after.analytical_direction == before.analytical_direction
    assert after.confidence < before.confidence
    assert after.confidence_decomposition.regime_suitability_penalty == 20


@pytest.mark.asyncio
async def test_execution_rejection_never_erases_analytical_direction() -> None:
    state, quant = await state_and_quant()
    result = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, aligned_analysis(state, quant)
    )
    assert all(item.execution_eligibility == ExecutionEligibility.INELIGIBLE for item in result.timeframe_signals)
    assert all(item.blocking_reasons for item in result.timeframe_signals)
    assert all(item.analytical_direction in {AnalyticalDirection.BUY, AnalyticalDirection.SELL} for item in result.timeframe_signals)


@pytest.mark.asyncio
async def test_repeated_cycle_persists_one_signal_set() -> None:
    state, quant = await state_and_quant()
    value = MultiTimeframeSignalSynthesizer().synthesize(
        state, quant, aligned_analysis(state, quant)
    )
    repository = InMemoryMultiTimeframeSignalRepository()
    first = await repository.save(value)
    second = await repository.save(value)
    assert first == second
    assert len(repository.values) == 1
