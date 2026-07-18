from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .models import SourceEvidence


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _direction(value: object) -> float:
    text = str(value).lower()
    if text in {"bullish", "up", "uptrend", "positive", "buy_side", "accumulation"}:
        return 1.0
    if text in {"bearish", "down", "downtrend", "negative", "sell_side", "distribution"}:
        return -1.0
    return 0.0


def normalized_source(
    source: str,
    group: str,
    features: Mapping[str, Any],
    timestamp: datetime,
    version: str,
    evidence_id: str,
) -> SourceEvidence:
    direction = 0.0
    confidence = 0.5
    quality = 0.5
    risk = 0.0
    reasons: tuple[str, ...] = ("normalized_upstream_evidence",)
    degraded = False
    if source == "market_regime":
        direction = _number(features.get("net_directional_score"), _direction(features.get("directional_bias")))
        confidence = _number(features.get("confidence"), 0.5)
        quality = _number(features.get("quality"), 0.5)
        conflict = _number(features.get("conflict"))
        risk = max(conflict, 0.75 if "high" in str(features.get("volatility", "")) else 0.0)
        reasons = (f"regime_{features.get('directional_bias', 'neutral')}",)
    elif source == "smc":
        direction = _direction(features.get("current_structure_direction"))
        confidence = _number(features.get("smc_confidence"), 0.5)
        quality = _number(features.get("smc_input_quality"), 0.5)
        reasons = (f"structure_{features.get('current_structure_direction', 'neutral')}",)
    elif source == "liquidity":
        above = _number(features.get("liquidity_density_above"))
        below = _number(features.get("liquidity_density_below"))
        direction = (above - below) / (above + below) if above + below else 0.0
        confidence = _number(features.get("confidence"), 0.5)
        quality = _number(features.get("data_quality"), 0.5)
        risk = 0.4 if features.get("latest_sweep") or features.get("latest_raid") else 0.1
        reasons = ("liquidity_distribution",)
    elif source == "volume_profile":
        migration = _mapping(features.get("poc_migration"))
        direction = _direction(migration.get("direction"))
        confidence = _number(features.get("confidence"), 0.5)
        quality_map = _mapping(features.get("data_quality"))
        quality = _number(quality_map.get("overall"), _number(features.get("volume_concentration"), 0.5))
        risk = 0.2 if features.get("active_gaps") else 0.05
        reasons = ("profile_context",)
    elif source == "institutional_flow":
        pressure = _mapping(features.get("directional_pressure"))
        direction = _number(pressure.get("net_pressure"))
        confidence = _number(pressure.get("confidence"), 0.5)
        quality_map = _mapping(features.get("quality"))
        quality = _number(quality_map.get("overall"), confidence)
        risk = _number(features.get("ambiguity"), _number(pressure.get("conflict")))
        reasons = ("directional_flow_pressure",)
    elif source == "economic_calendar":
        risk = _number(features.get("risk_score"))
        confidence = _number(features.get("relevance_score"), 0.5)
        quality = _number(features.get("quality_score"), 0.5)
        reasons = (f"event_risk_{features.get('risk_window_phase', 'outside')}",)
        degraded = bool(features.get("unavailable_context"))
    return SourceEvidence(
        source=source,
        source_group=group,
        source_version=version,
        evidence_id=evidence_id,
        source_timestamp=timestamp,
        observation_timestamp=timestamp,
        publication_timestamp=timestamp,
        direction=max(-1.0, min(1.0, direction)),
        confidence=max(0.0, min(1.0, confidence)),
        quality=max(0.0, min(1.0, quality)),
        risk=max(0.0, min(1.0, risk)),
        degraded=degraded,
        reason_codes=reasons,
    )
