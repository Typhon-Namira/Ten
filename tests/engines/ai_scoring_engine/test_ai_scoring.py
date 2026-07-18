from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from backend.app.engines.ai_scoring_engine import (
    AIScoringConfig,
    ComponentConfig,
    DeterministicAIScoringEngine,
    DirectionalLabel,
    FixedClock,
    FreshnessState,
    ScoreMode,
    ScoreStatus,
    ScoringInput,
    SourceEvidence,
)
from backend.app.engines.ai_scoring_engine.config import ConflictPolicy, LabelThresholds
from backend.app.engines.ai_scoring_engine.normalization import normalized_source

NOW = datetime(2026, 7, 19, 12, tzinfo=UTC)


def evidence(
    source: str,
    group: str,
    direction: float,
    *,
    age: timedelta = timedelta(),
    quality: float = 0.9,
    confidence: float = 0.8,
    risk: float = 0.1,
    degraded: bool = False,
) -> SourceEvidence:
    timestamp = NOW - age
    return SourceEvidence(
        source=source,
        source_group=group,
        source_version="1.0.0",
        evidence_id=f"{source}-1",
        source_timestamp=timestamp,
        observation_timestamp=timestamp,
        publication_timestamp=timestamp,
        direction=direction,
        confidence=confidence,
        quality=quality,
        risk=risk,
        degraded=degraded,
        reason_codes=(f"{source}_reason",),
    )


def scoring_input(*sources: SourceEvidence, mode: ScoreMode = ScoreMode.LIVE) -> ScoringInput:
    values = {item.source: item for item in sources}
    return ScoringInput(
        instrument="XAUUSD",
        timeframe="M15",
        as_of=NOW,
        requested_at=NOW,
        mode=mode,
        market_data=values.get("market_data"),
        market_regime=values.get("market_regime"),
        smc=values.get("smc"),
        liquidity=values.get("liquidity"),
        volume_profile=values.get("volume_profile"),
        institutional_flow=values.get("institutional_flow"),
        economic_calendar=values.get("economic_calendar"),
    )


def aligned_input(*, risk: float = 0.1, mode: ScoreMode = ScoreMode.LIVE) -> ScoringInput:
    return scoring_input(
        evidence("market_data", "data", 0, risk=risk),
        evidence("market_regime", "context", 0.7, risk=risk),
        evidence("smc", "structure", 0.8, risk=risk),
        evidence("liquidity", "structure", 0.5, risk=risk),
        evidence("volume_profile", "participation", 0.4, risk=risk),
        evidence("institutional_flow", "participation", 0.9, risk=risk),
        evidence("economic_calendar", "event_risk", 0, risk=risk),
        mode=mode,
    )


def test_deterministic_aligned_score_and_safety_invariants() -> None:
    engine = DeterministicAIScoringEngine(clock=FixedClock(NOW))
    first = engine.score(aligned_input())
    second = engine.score(aligned_input())
    assert first == second
    assert first.status == ScoreStatus.READY
    assert first.directional_score > 0
    assert first.directional_label in {DirectionalLabel.BULLISH, DirectionalLabel.STRONG_BULLISH}
    assert 0 <= first.confidence_score <= 100
    assert first.evidence_alignment_score == 100
    assert sum(item.directional_contribution for item in first.components) == pytest.approx(first.directional_score, abs=1e-3)
    assert first.metadata.trading_instruction is False
    assert first.metadata.order_execution is False
    assert first.explanation.financial_safety_code == "analytical_intelligence_only"
    assert first.metadata.input_fingerprint == second.metadata.input_fingerprint


def test_risk_reduces_composite_without_reversing_direction() -> None:
    engine = DeterministicAIScoringEngine(clock=FixedClock(NOW))
    low = engine.score(aligned_input(risk=0.0))
    high = engine.score(aligned_input(risk=1.0))
    assert low.directional_score == high.directional_score
    assert high.market_risk_score > low.market_risk_score
    assert 0 < high.composite_score < low.composite_score


def test_conflicts_are_structured_stable_and_penalize_confidence() -> None:
    engine = DeterministicAIScoringEngine(clock=FixedClock(NOW))
    aligned = engine.score(scoring_input(evidence("smc", "structure", 0.9), evidence("institutional_flow", "participation", 0.9)))
    conflict = engine.score(scoring_input(evidence("smc", "structure", 0.9), evidence("institutional_flow", "participation", -0.9)))
    assert conflict.conflicts[0].severity == "severe"
    assert conflict.conflicts[0].conflict_type == "cross_group_directional"
    assert conflict.confidence_score < aligned.confidence_score
    same_group = engine.score(scoring_input(evidence("smc", "structure", 0.8), evidence("liquidity", "structure", -0.8)))
    assert same_group.conflicts[0].conflict_type == "same_group_directional"


def test_missing_degraded_stale_and_replay_statuses() -> None:
    engine = DeterministicAIScoringEngine(clock=FixedClock(NOW))
    single = engine.score(scoring_input(evidence("smc", "structure", 1)))
    assert single.status == ScoreStatus.INSUFFICIENT_EVIDENCE
    assert single.confidence_score <= engine.config.evidence.single_source_confidence_ceiling
    degraded = engine.score(scoring_input(evidence("smc", "structure", 0.8, degraded=True), evidence("institutional_flow", "participation", 0.8)))
    assert degraded.status == ScoreStatus.DEGRADED
    assert degraded.confidence_score <= engine.config.evidence.degraded_confidence_ceiling
    stale = engine.score(scoring_input(evidence("smc", "structure", 0.8, age=timedelta(minutes=40)), evidence("institutional_flow", "participation", 0.8)))
    assert stale.status == ScoreStatus.STALE
    assert FreshnessState.STALE in {item.freshness_state for item in stale.components}
    replay = engine.score(aligned_input(mode=ScoreMode.REPLAY))
    assert replay.status == ScoreStatus.REPLAY
    aging = engine.score(scoring_input(evidence("smc", "structure", 0.8, age=timedelta(minutes=10)), evidence("institutional_flow", "participation", 0.8)))
    assert FreshnessState.AGING in {item.freshness_state for item in aging.components}


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (-100, DirectionalLabel.STRONG_BEARISH),
        (-70, DirectionalLabel.BEARISH),
        (-35, DirectionalLabel.SLIGHTLY_BEARISH),
        (-10, DirectionalLabel.NEUTRAL),
        (10, DirectionalLabel.NEUTRAL),
        (10.01, DirectionalLabel.SLIGHTLY_BULLISH),
        (35.01, DirectionalLabel.BULLISH),
        (70.01, DirectionalLabel.STRONG_BULLISH),
    ],
)
def test_directional_label_boundaries(score: float, label: DirectionalLabel) -> None:
    assert DeterministicAIScoringEngine().label(score) == label


def test_point_in_time_and_timestamp_validation() -> None:
    future = evidence("smc", "structure", 1).model_copy(update={"publication_timestamp": NOW + timedelta(seconds=1)})
    with pytest.raises(ValidationError, match="future evidence"):
        scoring_input(future)
    with pytest.raises(ValidationError, match="timezone-aware"):
        SourceEvidence(
            source="smc",
            source_group="structure",
            source_version="1",
            evidence_id="x",
            source_timestamp=datetime(2026, 1, 1),
            observation_timestamp=NOW,
            publication_timestamp=NOW,
            direction=0,
            confidence=0,
            quality=0,
            risk=0,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        FixedClock(datetime(2026, 1, 1))


def test_configuration_rejects_impossible_policies() -> None:
    with pytest.raises(ValidationError, match="fresh_seconds"):
        ComponentConfig(directional_weight=1, source_group="structure", fresh_seconds=10, stale_seconds=10)
    with pytest.raises(ValidationError, match="unknown source group"):
        ComponentConfig(directional_weight=1, source_group="unknown")
    with pytest.raises(ValidationError, match="conflict threshold"):
        ConflictPolicy(directional_gap_threshold=1.8, severe_gap_threshold=1.7)
    with pytest.raises(ValidationError, match="strictly increasing"):
        LabelThresholds(bearish=-70)
    with pytest.raises(ValidationError, match="every approved source"):
        AIScoringConfig(components={"smc": ComponentConfig(directional_weight=1, source_group="structure")})
    values = AIScoringConfig().components.copy()
    values = {name: item.model_copy(update={"directional_weight": 0}) for name, item in values.items()}
    with pytest.raises(ValidationError, match="directional component"):
        AIScoringConfig(components=values)


@pytest.mark.parametrize(
    ("source", "features", "expected_direction"),
    [
        ("market_regime", {"net_directional_score": -0.6, "confidence": 0.8, "quality": 0.9, "volatility": "high", "directional_bias": "bearish"}, -0.6),
        ("smc", {"current_structure_direction": "bullish", "smc_confidence": 0.8, "smc_input_quality": 0.9}, 1),
        ("liquidity", {"liquidity_density_above": 3, "liquidity_density_below": 1, "confidence": 0.8, "data_quality": 0.9, "latest_sweep": {}}, 0.5),
        ("volume_profile", {"poc_migration": {"direction": "down"}, "confidence": 0.8, "data_quality": {"overall": 0.7}, "active_gaps": [1]}, -1),
        ("institutional_flow", {"directional_pressure": {"net_pressure": 0.75, "confidence": 0.8}, "quality": {"overall": 0.9}, "ambiguity": 0.2}, 0.75),
        ("economic_calendar", {"risk_score": 0.9, "relevance_score": 0.8, "quality_score": 0.7, "risk_window_phase": "active", "unavailable_context": ["provider"]}, 0),
    ],
)
def test_approved_upstream_normalization(source: str, features: dict[str, object], expected_direction: float) -> None:
    group = AIScoringConfig().components[source].source_group
    item = normalized_source(source, group, features, NOW, "1.0.0", "evidence")
    assert item.direction == expected_direction
    assert 0 <= item.confidence <= 1
    assert 0 <= item.quality <= 1
    assert 0 <= item.risk <= 1


def test_non_numeric_normalization_is_bounded_and_neutral() -> None:
    item = normalized_source("liquidity", "structure", {"liquidity_density_above": True, "liquidity_density_below": "bad"}, NOW, "1", "x")
    assert item.direction == 0
    assert item.confidence == 0.5
