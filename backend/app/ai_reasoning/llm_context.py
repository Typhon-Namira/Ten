"""Strict, bounded DTOs for the external LLM boundary.

Internal analytical and persistence models are intentionally accepted only by the
deterministic builder below.  The OpenRouter adapter serializes ``LLMAnalysisContext``
and never serializes those internal objects directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import AIReasoningRequest

_MAX_SUMMARY_CHARS = 480
_MAX_REASON_CHARS = 160


class _ImmutableDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeframeTrendSummary(_ImmutableDTO):
    timeframe: Literal["M1", "M5", "M15"]
    state: str = Field(max_length=48)
    stale: bool
    freshness_seconds: float = Field(ge=0)
    summary: str = Field(max_length=_MAX_SUMMARY_CHARS)


class CompactEngineSummary(_ImmutableDTO):
    status: str = Field(max_length=48)
    summary: str = Field(max_length=_MAX_SUMMARY_CHARS)
    evidence_ids: tuple[UUID, ...] = Field(default=(), max_length=5)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=5)

    @field_validator("reason_codes")
    @classmethod
    def bounded_reasons(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(value[:_MAX_REASON_CHARS] for value in values)


class CompactPriceLevel(_ImmutableDTO):
    kind: str = Field(max_length=48)
    price: float = Field(gt=0)
    timeframe: str | None = Field(default=None, max_length=8)
    evidence_id: UUID | None = None


class CompactZone(_ImmutableDTO):
    kind: str = Field(max_length=48)
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)
    timeframe: str | None = Field(default=None, max_length=8)
    evidence_id: UUID | None = None

    @model_validator(mode="after")
    def ordered(self) -> CompactZone:
        if self.lower > self.upper:
            raise ValueError("compact zone must be ordered")
        return self


class CompactVolumeProfile(_ImmutableDTO):
    status: str = Field(max_length=48)
    poc: float | None = Field(default=None, gt=0)
    value_area_high: float | None = Field(default=None, gt=0)
    value_area_low: float | None = Field(default=None, gt=0)
    nearest_hvns: tuple[float, ...] = Field(default=(), max_length=3)
    nearest_lvns: tuple[float, ...] = Field(default=(), max_length=3)
    shape: str | None = Field(default=None, max_length=64)
    summary: str = Field(max_length=_MAX_SUMMARY_CHARS)


class CompactQuantForecast(_ImmutableDTO):
    status: str = Field(max_length=48)
    horizon: str | None = Field(default=None, max_length=24)
    dominant_direction: Literal["BUY", "SELL", "NEUTRAL"]
    buy_probability: float = Field(ge=0, le=1)
    sell_probability: float = Field(ge=0, le=1)
    neutral_probability: float = Field(ge=0, le=1)
    expected_return: float | None = None
    expected_minimum_movement: float | None = Field(default=None, ge=0)
    expected_base_movement: float | None = Field(default=None, ge=0)
    expected_maximum_movement: float | None = Field(default=None, ge=0)
    expected_volatility: float | None = Field(default=None, ge=0)
    expected_favorable_excursion: float | None = Field(default=None, ge=0)
    expected_adverse_excursion: float | None = Field(default=None, ge=0)
    tp1_probability: float | None = Field(default=None, ge=0, le=1)
    tp2_probability: float | None = Field(default=None, ge=0, le=1)
    sl_before_tp_probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> CompactQuantForecast:
        if abs(self.buy_probability + self.sell_probability + self.neutral_probability - 1) > 1e-8:
            raise ValueError("compact Quant probabilities must sum to one")
        return self


class CompactRiskSummary(_ImmutableDTO):
    data_quality_status: str = Field(max_length=48)
    evidence_completeness: float = Field(ge=0, le=1)
    missing_evidence_count: int = Field(ge=0)
    degraded_evidence_count: int = Field(ge=0)
    stale_evidence_count: int = Field(ge=0)
    spread: float | None = Field(default=None, ge=0)
    flags: tuple[str, ...] = Field(default=(), max_length=5)

    @field_validator("flags")
    @classmethod
    def bounded_flags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(value[:_MAX_REASON_CHARS] for value in values)


class CompactPositionSummary(_ImmutableDTO):
    signal_id: UUID
    state: str = Field(max_length=48)
    direction: Literal["BUY", "SELL", "NEUTRAL"]
    setup_family: str = Field(max_length=64)
    entry_low: float | None = Field(default=None, gt=0)
    entry_high: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit_levels: tuple[float, ...] = Field(default=(), max_length=3)


class PreviousDecisionSummary(_ImmutableDTO):
    decision: Literal["BUY", "SELL", "NEUTRAL", "WAIT"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = Field(default=None, max_length=_MAX_SUMMARY_CHARS)
    generated_at: datetime | None = None

    @field_validator("generated_at")
    @classmethod
    def aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("previous decision timestamp must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None


class MaterialChange(_ImmutableDTO):
    category: str = Field(max_length=48)
    summary: str = Field(max_length=_MAX_SUMMARY_CHARS)


class LLMAnalysisContext(_ImmutableDTO):
    """The only market-analysis object allowed to cross the LLM boundary."""

    schema_version: Literal["2.0"] = "2.0"
    request_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    quantitative_forecast_id: UUID
    symbol: str = Field(max_length=32)
    analysis_boundary: datetime
    market_data_cutoff: datetime
    current_price: float = Field(gt=0)
    market_regime: CompactEngineSummary
    timeframe_trends: tuple[TimeframeTrendSummary, ...] = Field(max_length=3)
    smc: CompactEngineSummary
    nearest_supply_zones: tuple[CompactZone, ...] = Field(default=(), max_length=3)
    nearest_demand_zones: tuple[CompactZone, ...] = Field(default=(), max_length=3)
    relevant_order_blocks: tuple[CompactZone, ...] = Field(default=(), max_length=3)
    relevant_fair_value_gaps: tuple[CompactZone, ...] = Field(default=(), max_length=3)
    nearest_liquidity_levels: tuple[CompactPriceLevel, ...] = Field(default=(), max_length=5)
    volume_profile: CompactVolumeProfile
    institutional_flow: CompactEngineSummary
    quant: CompactQuantForecast
    risk: CompactRiskSummary
    active_position: CompactPositionSummary | None = None
    previous_final_decision: PreviousDecisionSummary | None = None
    material_changes: tuple[MaterialChange, ...] = Field(default=(), max_length=5)
    prompt_version: str = Field(max_length=64)
    reasoning_policy_version: str = Field(max_length=64)
    setup_family_registry_version: str = Field(max_length=64)
    model_identifier: str = Field(max_length=128)

    @field_validator("analysis_boundary", "market_data_cutoff")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("LLM context timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def point_in_time_boundary(self) -> LLMAnalysisContext:
        if self.analysis_boundary > self.market_data_cutoff:
            raise ValueError("LLM context exceeds its market-data cutoff")
        return self


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk(nested)


def _clean_scalar(value: Any) -> str | None:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        return text[:96] or None
    return None


def _facts(value: Any, *, maximum: int = 8) -> tuple[str, ...]:
    preferred = (
        "status",
        "regime",
        "direction",
        "bias",
        "trend",
        "classification",
        "shape",
        "session",
        "phase",
        "volatility",
        "source_status",
    )
    found: list[str] = []
    for mapping in _walk(value):
        for key in preferred:
            if key in mapping:
                scalar = _clean_scalar(mapping[key])
                fact = f"{key}={scalar}" if scalar is not None else None
                if fact and fact not in found:
                    found.append(fact)
                    if len(found) >= maximum:
                        return tuple(found)
    return tuple(found)


def _evidence_items(request: AIReasoningRequest, *field_names: str) -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    for field_name in field_names:
        value = getattr(request, field_name, ())
        if isinstance(value, tuple):
            items.extend(item for item in value if isinstance(item, dict))
    return tuple(items)


def _engine_summary(items: tuple[dict[str, Any], ...], *, fallback: str) -> CompactEngineSummary:
    statuses = tuple(str(item.get("availability", "unknown")) for item in items)
    status = "degraded" if any(value in {"degraded", "stale", "unavailable"} for value in statuses) else (
        "available" if items else "unavailable"
    )
    facts: list[str] = []
    evidence_ids: list[UUID] = []
    reasons: list[str] = []
    for item in items:
        raw = item.get("raw")
        for fact in _facts(raw):
            if fact not in facts:
                facts.append(fact)
        try:
            evidence_id = UUID(str(item["evidence_id"]))
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        except (KeyError, TypeError, ValueError):
            pass
        for reason in item.get("reason_codes", ()):
            text = str(reason)[:_MAX_REASON_CHARS]
            if text not in reasons:
                reasons.append(text)
    summary = "; ".join(facts[:8]) or fallback
    return CompactEngineSummary(
        status=status,
        summary=summary[:_MAX_SUMMARY_CHARS],
        evidence_ids=tuple(evidence_ids[:5]),
        reason_codes=tuple(reasons[:5]),
    )


def _candidate_zones(items: tuple[dict[str, Any], ...], current_price: float) -> tuple[CompactZone, ...]:
    candidates: list[CompactZone] = []
    for item in items:
        evidence_id: UUID | None
        try:
            evidence_id = UUID(str(item.get("evidence_id")))
        except (TypeError, ValueError):
            evidence_id = None
        timeframe = str(item.get("timeframe"))[:8] if item.get("timeframe") else None
        for mapping in _walk(item.get("raw")):
            lower = next(
                (mapping.get(key) for key in ("lower_price", "lower", "low", "start_price") if isinstance(mapping.get(key), (int, float))),
                None,
            )
            upper = next(
                (mapping.get(key) for key in ("upper_price", "upper", "high", "end_price") if isinstance(mapping.get(key), (int, float))),
                None,
            )
            if lower is None or upper is None or float(lower) <= 0 or float(upper) <= 0:
                continue
            kind = next(
                (
                    str(mapping[key])
                    for key in ("zone_type", "type", "kind", "side", "direction", "classification")
                    if mapping.get(key) is not None
                ),
                "zone",
            )[:48]
            try:
                zone = CompactZone(
                    kind=kind,
                    lower=min(float(lower), float(upper)),
                    upper=max(float(lower), float(upper)),
                    timeframe=timeframe,
                    evidence_id=evidence_id,
                )
            except ValueError:
                continue
            if (zone.kind, zone.lower, zone.upper, zone.evidence_id) not in {
                (existing.kind, existing.lower, existing.upper, existing.evidence_id) for existing in candidates
            }:
                candidates.append(zone)
    return tuple(sorted(candidates, key=lambda zone: abs(((zone.lower + zone.upper) / 2) - current_price)))


def _candidate_levels(items: tuple[dict[str, Any], ...], current_price: float) -> tuple[CompactPriceLevel, ...]:
    candidates: list[CompactPriceLevel] = []
    for item in items:
        try:
            evidence_id = UUID(str(item.get("evidence_id")))
        except (TypeError, ValueError):
            evidence_id = None
        timeframe = str(item.get("timeframe"))[:8] if item.get("timeframe") else None
        for mapping in _walk(item.get("raw")):
            price = next(
                (
                    mapping.get(key)
                    for key in ("price", "level_price", "peak_price", "midpoint")
                    if isinstance(mapping.get(key), (int, float)) and float(mapping[key]) > 0
                ),
                None,
            )
            if price is None:
                continue
            kind = next(
                (
                    str(mapping[key])
                    for key in ("level_type", "type", "kind", "classification", "side")
                    if mapping.get(key) is not None
                ),
                "level",
            )[:48]
            candidate = CompactPriceLevel(
                kind=kind,
                price=float(price),
                timeframe=timeframe,
                evidence_id=evidence_id,
            )
            if (candidate.kind, candidate.price, candidate.evidence_id) not in {
                (existing.kind, existing.price, existing.evidence_id) for existing in candidates
            }:
                candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda level: abs(level.price - current_price)))


def _number_for_keys(value: Any, keys: tuple[str, ...]) -> float | None:
    for mapping in _walk(value):
        for key in keys:
            candidate = mapping.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and float(candidate) > 0:
                return float(candidate)
            if isinstance(candidate, dict):
                nested = next(
                    (
                        candidate.get(nested_key)
                        for nested_key in ("price", "peak_price", "value")
                        if isinstance(candidate.get(nested_key), (int, float))
                    ),
                    None,
                )
                if nested is not None and float(nested) > 0:
                    return float(nested)
    return None


def _numbers_for_token(value: Any, token: str, current_price: float) -> tuple[float, ...]:
    values: list[float] = []
    for mapping in _walk(value):
        kind = " ".join(
            str(mapping.get(key, "")).lower()
            for key in ("type", "kind", "node_type", "classification")
        )
        if token not in kind:
            continue
        price = next(
            (
                mapping.get(key)
                for key in ("peak_price", "price", "midpoint")
                if isinstance(mapping.get(key), (int, float)) and float(mapping[key]) > 0
            ),
            None,
        )
        if price is not None and float(price) not in values:
            values.append(float(price))
    return tuple(sorted(values, key=lambda price: abs(price - current_price))[:3])


def _position(request: AIReasoningRequest) -> CompactPositionSummary | None:
    value = request.existing_signal_state
    if not isinstance(value, dict):
        return None
    try:
        signal_id = UUID(str(value["signal_id"]))
    except (KeyError, TypeError, ValueError):
        return None
    entry_value = value.get("entry_zone")
    entry: dict[str, Any] = entry_value if isinstance(entry_value, dict) else {}
    direction = str(value.get("direction", "NEUTRAL")).upper()
    if direction not in {"BUY", "SELL", "NEUTRAL"}:
        direction = "NEUTRAL"
    targets = tuple(
        float(item)
        for item in value.get("take_profit_levels", ())
        if isinstance(item, (int, float)) and float(item) > 0
    )[:3]
    return CompactPositionSummary(
        signal_id=signal_id,
        state=str(value.get("state", "unknown"))[:48],
        direction=direction,
        setup_family=str(value.get("setup_family", "unknown"))[:64],
        entry_low=float(entry["low"]) if isinstance(entry.get("low"), (int, float)) else None,
        entry_high=float(entry["high"]) if isinstance(entry.get("high"), (int, float)) else None,
        stop_loss=float(value["stop_loss"]) if isinstance(value.get("stop_loss"), (int, float)) else None,
        take_profit_levels=targets,
    )


def _previous_decision(request: AIReasoningRequest) -> PreviousDecisionSummary | None:
    value = request.previous_ai_forecast
    if not isinstance(value, dict):
        return None
    decision = str(value.get("dominant_direction", "WAIT")).upper()
    if decision not in {"BUY", "SELL", "NEUTRAL", "WAIT"}:
        decision = "WAIT"
    confidence = next(
        (
            float(value[key])
            for key in ("forecast_confidence", "dominant_scenario_probability")
            if isinstance(value.get(key), (int, float))
        ),
        None,
    )
    generated = value.get("generated_at")
    if isinstance(generated, str):
        try:
            generated = datetime.fromisoformat(generated.replace("Z", "+00:00"))
        except ValueError:
            generated = None
    return PreviousDecisionSummary(
        decision=decision,
        confidence=confidence,
        reason=str(value.get("reasoning_summary"))[:_MAX_SUMMARY_CHARS]
        if value.get("reasoning_summary")
        else None,
        generated_at=generated if isinstance(generated, datetime) else None,
    )


def build_llm_analysis_context(request: AIReasoningRequest) -> LLMAnalysisContext:
    """Deterministically reduce an internal request to the strict external DTO."""

    probabilities = {
        key: float(value or 0)
        for key, value in request.quantitative_probabilities.items()
        if key in {"BUY", "SELL", "NEUTRAL"}
    }
    probabilities = {
        "BUY": probabilities.get("BUY", 0.0),
        "SELL": probabilities.get("SELL", 0.0),
        "NEUTRAL": probabilities.get("NEUTRAL", 0.0),
    }
    total = sum(probabilities.values())
    if total <= 0:
        probabilities = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 1.0}
    else:
        probabilities = {key: value / total for key, value in probabilities.items()}
    dominant = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]

    reference_price = request.current_price

    smc_items = _evidence_items(request, "smc_evidence")
    liquidity_items = _evidence_items(request, "liquidity_pools")
    volume_items = _evidence_items(request, "volume_profile_evidence")
    regime_items = _evidence_items(request, "market_regime")
    flow_items = _evidence_items(request, "institutional_flow_evidence")
    zones = _candidate_zones(smc_items, reference_price)
    levels = _candidate_levels(liquidity_items, reference_price)

    supply_tokens = ("supply", "bearish", "resistance", "sell")
    demand_tokens = ("demand", "bullish", "support", "buy")
    supply = tuple(zone for zone in zones if any(token in zone.kind.lower() for token in supply_tokens))[:3]
    demand = tuple(zone for zone in zones if any(token in zone.kind.lower() for token in demand_tokens))[:3]
    order_blocks = tuple(zone for zone in zones if "order" in zone.kind.lower() or "block" in zone.kind.lower())[:3]
    fair_value_gaps = tuple(zone for zone in zones if "fvg" in zone.kind.lower() or "fair" in zone.kind.lower())[:3]

    volume_raw = tuple(item.get("raw") for item in volume_items)
    volume_facts = "; ".join(fact for value in volume_raw for fact in _facts(value))[:_MAX_SUMMARY_CHARS]
    volume_status = "available" if volume_items else "unavailable"
    if any(str(item.get("availability")) in {"degraded", "stale", "unavailable"} for item in volume_items):
        volume_status = "degraded"

    trend_by_timeframe: list[TimeframeTrendSummary] = []
    trend_evidence = _evidence_items(request, "trend_evidence")
    for item in request.supported_timeframe_states[:3]:
        timeframe = str(item.get("timeframe", "M1"))
        if timeframe not in {"M1", "M5", "M15"}:
            continue
        facts = [
            fact
            for evidence in trend_evidence
            if evidence.get("timeframe") == timeframe
            for fact in _facts(evidence.get("raw"))
        ]
        trend_by_timeframe.append(
            TimeframeTrendSummary(
                timeframe=timeframe,
                state="stale" if bool(item.get("stale")) else "available",
                stale=bool(item.get("stale")),
                freshness_seconds=float(item.get("freshness_seconds", 0)),
                summary=("; ".join(dict.fromkeys(facts)) or "No compact directional fact available")[
                    :_MAX_SUMMARY_CHARS
                ],
            )
        )

    memory_changes: list[MaterialChange] = []
    memory_fields = (
        ("regime", request.market_memory.regime_transitions),
        ("structure", request.market_memory.structure_changes),
        ("liquidity", request.market_memory.liquidity_events),
        ("forecast", request.market_memory.forecast_changes),
        ("evidence", request.market_memory.evidence_changes),
        ("signal", request.market_memory.signal_state_changes),
    )
    for category, values in memory_fields:
        for value in values:
            memory_changes.append(
                MaterialChange(category=category, summary=str(value)[:_MAX_SUMMARY_CHARS])
            )

    risk_flags = tuple(
        dict.fromkeys(
            [
                *(f"missing:{value}" for value in request.missing_evidence[:5]),
                *(f"degraded:{value}" for value in request.degraded_evidence[:5]),
                *(f"stale:{value}" for value in request.stale_evidence[:5]),
            ]
        )
    )[:5]

    return LLMAnalysisContext(
        request_id=request.request_id,
        cycle_id=request.cycle_id,
        market_state_id=request.market_state_id,
        quantitative_forecast_id=request.quantitative_forecast_id,
        symbol=request.instrument,
        analysis_boundary=request.analysis_timestamp,
        market_data_cutoff=request.knowledge_cutoff,
        current_price=reference_price,
        market_regime=_engine_summary(regime_items, fallback="Market regime unavailable"),
        timeframe_trends=tuple(trend_by_timeframe),
        smc=_engine_summary(smc_items, fallback="SMC summary unavailable"),
        nearest_supply_zones=supply,
        nearest_demand_zones=demand,
        relevant_order_blocks=order_blocks,
        relevant_fair_value_gaps=fair_value_gaps,
        nearest_liquidity_levels=levels[:5],
        volume_profile=CompactVolumeProfile(
            status=volume_status,
            poc=_number_for_keys(volume_raw, ("poc", "developing_poc")),
            value_area_high=_number_for_keys(volume_raw, ("vah", "value_area_high")),
            value_area_low=_number_for_keys(volume_raw, ("val", "value_area_low")),
            nearest_hvns=_numbers_for_token(volume_raw, "hvn", reference_price),
            nearest_lvns=_numbers_for_token(volume_raw, "lvn", reference_price),
            shape=next((fact.split("=", 1)[1] for value in volume_raw for fact in _facts(value) if fact.startswith("shape=")), None),
            summary=volume_facts or "Volume Profile summary unavailable",
        ),
        institutional_flow=_engine_summary(flow_items, fallback="Institutional Flow summary unavailable"),
        quant=CompactQuantForecast(
            status=str(request.data_quality_summary.get("quant_status", "unknown"))[:48],
            horizon="10_m1",
            dominant_direction=dominant,
            buy_probability=probabilities["BUY"],
            sell_probability=probabilities["SELL"],
            neutral_probability=probabilities["NEUTRAL"],
            expected_return=request.expected_return,
            expected_minimum_movement=request.expected_movement.get("minimum"),
            expected_base_movement=request.expected_movement.get("base"),
            expected_maximum_movement=request.expected_movement.get("maximum"),
            expected_volatility=request.expected_volatility,
            expected_favorable_excursion=request.expected_favorable_excursion,
            expected_adverse_excursion=request.expected_adverse_excursion,
            tp1_probability=request.tp_probabilities.get("TP1"),
            tp2_probability=request.tp_probabilities.get("TP2"),
            sl_before_tp_probability=request.sl_before_tp_probability,
        ),
        risk=CompactRiskSummary(
            data_quality_status=str(request.data_quality_summary.get("state_status", "unknown"))[:48],
            evidence_completeness=float(request.data_quality_summary.get("evidence_completeness", 0)),
            missing_evidence_count=len(request.missing_evidence),
            degraded_evidence_count=len(request.degraded_evidence),
            stale_evidence_count=len(request.stale_evidence),
            spread=request.spread,
            flags=risk_flags,
        ),
        active_position=_position(request),
        previous_final_decision=_previous_decision(request),
        material_changes=tuple(memory_changes[-5:]),
        prompt_version=request.prompt_version,
        reasoning_policy_version=request.reasoning_policy_version,
        setup_family_registry_version=request.setup_family_registry_version,
        model_identifier=request.model_identifier,
    )


def compact_request_audit_payload(
    request: AIReasoningRequest,
    context: LLMAnalysisContext,
) -> dict[str, Any]:
    """Persist request identity and a fingerprintable compact contract, never engine payloads."""

    return {
        "schema_version": context.schema_version,
        "request_id": str(request.request_id),
        "cycle_id": str(request.cycle_id),
        "market_state_id": str(request.market_state_id),
        "quantitative_forecast_id": str(request.quantitative_forecast_id),
        "instrument": request.instrument,
        "analysis_timestamp": request.analysis_timestamp.isoformat(),
        "knowledge_cutoff": request.knowledge_cutoff.isoformat(),
        "prompt_version": request.prompt_version,
        "reasoning_policy_version": request.reasoning_policy_version,
        "model_identifier": request.model_identifier,
        "context": context.model_dump(mode="json"),
    }
