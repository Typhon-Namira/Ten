"""Build bounded, versioned AI reasoning requests from Phase 1/2 records."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.market_state import EvidenceItem, UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult

from .config import AIReasoningConfig
from .models import AIReasoningRequest, AIMarketForecast, AISignalProposal, ManagedSignal, MarketMemorySummary


_MAX_AI_COLLECTION_ITEMS = 1
_MAX_AI_STRING_CHARACTERS = 2_000


def _bounded_value(value: Any) -> Any:
    """Keep analytical summaries while bounding provider request size.

    Unified Market State remains the complete authoritative record. Engines can legitimately
    persist thousands of historical zones/levels in one snapshot; serializing those collections
    repeatedly into the LLM request produced payloads larger than the provider context. Long
    collections are represented by a deterministic latest sample and their exact count. The
    enclosing evidence ID already commits to the complete unabridged value. Every scalar and
    every top-level engine field remains present.
    """

    if isinstance(value, dict):
        return {str(key): _bounded_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= _MAX_AI_COLLECTION_ITEMS:
            return [_bounded_value(item) for item in value]
        return {
            "collection_summary": {
                "total_count": len(value),
                "sampling": "latest",
            },
            "items": [_bounded_value(value[-1])],
        }
    if isinstance(value, str) and len(value) > _MAX_AI_STRING_CHARACTERS:
        return {
            "text_summary": {
                "character_count": len(value),
            },
            "text": value[:_MAX_AI_STRING_CHARACTERS],
        }
    return value


def _evidence(item: EvidenceItem, *, bounded_raw: Any | None = None) -> dict[str, Any]:
    return {
        "evidence_id": str(item.evidence_id),
        "engine": item.source_engine,
        "timeframe": item.source_timeframe,
        "availability": item.availability.value,
        "confidence": item.confidence,
        "quality": item.quality,
        "uncertainty": item.uncertainty,
        "raw": _bounded_value(item.raw_value) if bounded_raw is None else bounded_raw,
        "reason_codes": item.reason_codes,
        "observed_at": item.observed_at.isoformat(),
        "available_at": item.available_at.isoformat(),
    }


def _evidence_reference(item: EvidenceItem) -> dict[str, Any]:
    return {
        "evidence_id": str(item.evidence_id),
        "engine": item.source_engine,
        "timeframe": item.source_timeframe,
        "availability": item.availability.value,
        "confidence": item.confidence,
        "quality": item.quality,
        "uncertainty": item.uncertainty,
        "reason_codes": item.reason_codes,
        "observed_at": item.observed_at.isoformat(),
        "available_at": item.available_at.isoformat(),
    }


class AIReasoningRequestBuilder:
    def __init__(
        self,
        config: AIReasoningConfig,
        *,
        model_identifier: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.model_identifier = model_identifier
        self.clock = clock or (lambda: datetime.now(UTC))

    def build(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        memory: MarketMemorySummary,
        *,
        existing_signal: ManagedSignal | None,
        previous_forecast: AIMarketForecast | None,
        previous_proposal: AISignalProposal | None,
    ) -> AIReasoningRequest:
        if quant.market_state_id != state.state_id or quant.point_in_time != state.market_data_boundary:
            raise ValueError("quantitative forecast does not belong to the immutable market state")
        prompt_version = (
            self.config.prompt_version_existing_signal
            if existing_signal is not None
            else self.config.prompt_version_new_market
        )
        request_id = uuid5(
            NAMESPACE_URL,
            f"ten:ai-reasoning:{state.state_id}:{quant.result_id}:{prompt_version}:{self.model_identifier}",
        )
        evidence = tuple(state.evidence)
        bounded_raw = {
            item.evidence_id: _bounded_value(item.raw_value)
            for item in evidence
        }
        grouped = {
            name: tuple(
                _evidence(item, bounded_raw=bounded_raw[item.evidence_id])
                for item in evidence
                if item.source_engine == name
            )
            for name in {
                "market_data",
                "smc",
                "liquidity",
                "volume_profile",
                "institutional_flow",
                "market_regime",
                "economic_calendar",
            }
        }
        raw_text = {
            item.evidence_id: str(bounded_raw[item.evidence_id]).lower()
            for item in evidence
        }

        def containing(*tokens: str) -> tuple[dict[str, Any], ...]:
            return tuple(
                _evidence_reference(item)
                for item in evidence
                if any(token in raw_text[item.evidence_id] for token in tokens)
            )

        prediction = next(
            (item for item in quant.predictions if item.horizon.horizon_id == "10_m1"),
            quant.predictions[0] if quant.predictions else None,
        )
        spread_value = next(
            (
                item.raw_value.get("spread")
                for item in evidence
                if item.source_engine == "market_data"
                and isinstance(item.raw_value, dict)
                and isinstance(item.raw_value.get("spread"), (int, float))
            ),
            None,
        )
        return AIReasoningRequest(
            request_id=request_id,
            cycle_id=state.cycle_id,
            market_state_id=state.state_id,
            quantitative_forecast_id=quant.result_id,
            instrument=state.instrument,
            analysis_timestamp=state.market_data_boundary,
            knowledge_cutoff=state.knowledge_cutoff,
            trigger_timeframe=state.trigger_timeframe,
            current_price=prediction.reference_price if prediction else 1.0,
            supported_timeframe_states=tuple(item.model_dump(mode="json") for item in state.timeframes),
            market_regime=grouped["market_regime"],
            trend_evidence=tuple(
                _evidence_reference(item)
                for item in evidence
                if item.source_engine in {"market_data", "smc"}
            ),
            volatility_evidence=tuple(
                _evidence_reference(item)
                for item in evidence
                if item.source_engine in {"market_data", "market_regime"}
            ),
            momentum_evidence=tuple(
                _evidence_reference(item)
                for item in evidence
                if item.source_engine == "institutional_flow"
            )
            + containing("momentum", "displacement"),
            structure_evidence=tuple(
                _evidence_reference(item)
                for item in evidence
                if item.source_engine == "smc"
            ),
            smc_evidence=grouped["smc"],
            bos_choch_mss_evidence=containing("bos", "choch", "mss"),
            liquidity_pools=grouped["liquidity"],
            liquidity_sweeps_and_raids=containing("sweep", "raid"),
            order_blocks=containing("order_block"),
            fair_value_gaps=containing("fvg", "fair_value_gap"),
            displacement_evidence=containing("displacement"),
            volume_profile_evidence=grouped["volume_profile"],
            poc_hvn_lvn_evidence=containing("poc", "hvn", "lvn"),
            institutional_flow_evidence=grouped["institutional_flow"],
            session_context=containing("session"),
            spread=float(spread_value) if spread_value is not None else None,
            economic_event_context=grouped["economic_calendar"],
            data_quality_summary={
                "state_status": state.status.value,
                "evidence_completeness": state.evidence_completeness,
                "quant_status": quant.status.value,
            },
            missing_evidence=state.unavailable_evidence,
            degraded_evidence=state.degraded_evidence,
            stale_evidence=state.stale_evidence,
            quantitative_probabilities={
                "BUY": prediction.buy_probability if prediction else None,
                "SELL": prediction.sell_probability if prediction else None,
                "NEUTRAL": prediction.neutral_probability if prediction else None,
            },
            expected_return=prediction.expected_return if prediction else None,
            expected_movement={
                "minimum": prediction.expected_minimum_movement if prediction else None,
                "base": prediction.expected_base_movement if prediction else None,
                "maximum": prediction.expected_maximum_movement if prediction else None,
            },
            expected_volatility=prediction.expected_volatility if prediction else None,
            expected_favorable_excursion=prediction.expected_mfe if prediction else None,
            expected_adverse_excursion=prediction.expected_mae if prediction else None,
            tp_probabilities={
                "TP1": prediction.tp1_probability if prediction else None,
                "TP2": prediction.tp2_probability if prediction else None,
            },
            sl_before_tp_probability=prediction.sl_before_tp_probability if prediction else None,
            market_memory=memory,
            existing_signal_state=existing_signal.model_dump(mode="json") if existing_signal else None,
            previous_ai_forecast=previous_forecast.model_dump(mode="json") if previous_forecast else None,
            previous_ai_proposal=previous_proposal.model_dump(mode="json") if previous_proposal else None,
            prompt_version=prompt_version,
            reasoning_policy_version=self.config.reasoning_policy_version,
            setup_family_registry_version=self.config.setup_family_registry_version,
            model_identifier=self.model_identifier,
            quantitative_model_version=quant.model_version,
            feature_schema_version=quant.feature_schema_version,
            market_state_schema_version=state.schema_version,
            created_at=max(self.clock(), state.market_data_boundary),
        )
