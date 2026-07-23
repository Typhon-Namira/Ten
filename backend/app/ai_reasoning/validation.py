"""Deterministic validation of untrusted LLM structured output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult

from .models import AIMarketForecast, AIReasoningRequest, AISignalProposal, AIResultStatus, ProposalAction
from .setup_families import SetupFamilyRegistry


def structural_opportunity_key(
    instrument: str,
    setup_family: str,
    direction: str,
    structural_anchor_ids: tuple[str, ...],
    scenario_identity: str,
) -> str:
    material = "|".join(
        (
            instrument.upper(),
            setup_family,
            direction,
            ",".join(sorted(structural_anchor_ids)),
            scenario_identity.strip().lower(),
        )
    )
    return sha256(material.encode()).hexdigest()


@dataclass(frozen=True)
class ValidatedAIOutput:
    forecast: AIMarketForecast
    proposal: AISignalProposal | None


class StructuredAIOutputError(ValueError):
    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class StructuredAIOutputValidator:
    def __init__(self, registry: SetupFamilyRegistry) -> None:
        self.registry = registry

    def validate(
        self,
        raw: dict[str, Any],
        *,
        request: AIReasoningRequest,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
    ) -> ValidatedAIOutput:
        errors: list[str] = []
        if not isinstance(raw.get("forecast"), dict):
            raise StructuredAIOutputError(("forecast_object_missing",))
        try:
            forecast = AIMarketForecast.model_validate(raw["forecast"])
        except ValidationError as exc:
            raise StructuredAIOutputError(tuple(f"forecast:{item['loc']}:{item['type']}" for item in exc.errors())) from exc
        proposal: AISignalProposal | None = None
        if raw.get("proposal") is not None:
            if not isinstance(raw["proposal"], dict):
                errors.append("proposal_object_invalid")
            else:
                try:
                    proposal = AISignalProposal.model_validate(raw["proposal"])
                except ValidationError as exc:
                    errors.extend(f"proposal:{item['loc']}:{item['type']}" for item in exc.errors())
        if errors:
            raise StructuredAIOutputError(tuple(errors))

        if forecast.request_id != request.request_id:
            errors.append("forecast_request_id_mismatch")
        if forecast.market_state_id != state.state_id or forecast.market_state_id != request.market_state_id:
            errors.append("forecast_market_state_id_mismatch")
        if forecast.quantitative_forecast_id != quant.result_id:
            errors.append("forecast_quantitative_id_mismatch")
        if forecast.cycle_id != state.cycle_id:
            errors.append("forecast_cycle_id_mismatch")
        if forecast.model_identifier != request.model_identifier:
            errors.append("forecast_model_identifier_mismatch")
        if forecast.prompt_version != request.prompt_version:
            errors.append("forecast_prompt_version_mismatch")
        if forecast.reasoning_policy_version != request.reasoning_policy_version:
            errors.append("forecast_policy_version_mismatch")
        if forecast.setup_family_registry_version != self.registry.version:
            errors.append("forecast_registry_version_mismatch")
        if forecast.generated_at < request.analysis_timestamp:
            errors.append("forecast_timestamp_before_analysis")
        if forecast.generated_at > request.created_at + timedelta(minutes=5):
            errors.append("forecast_timestamp_implausibly_future")

        valid_evidence_ids = {item.evidence_id for item in state.evidence}
        referenced = set(forecast.supporting_evidence_ids) | set(forecast.contradicting_evidence_ids)
        if not referenced <= valid_evidence_ids:
            errors.append("forecast_unknown_evidence_reference")
        valid_horizons = {item.horizon.horizon_id for item in quant.predictions}
        if forecast.expected_horizon is not None and forecast.expected_horizon not in valid_horizons:
            errors.append("forecast_unsupported_horizon")
        available_kinds = self._available_evidence_kinds(state)
        if forecast.selected_setup_family is not None:
            errors.extend(
                self.registry.validate_requirements(
                    forecast.selected_setup_family,
                    available_kinds,
                    forecast.evidence_completeness or 0,
                    proposal.recommended_action.value if proposal else ProposalAction.WAIT.value,
                )
            )
        elif forecast.status == AIResultStatus.AVAILABLE and proposal is not None and proposal.recommended_action != ProposalAction.WAIT:
            errors.append("actionable_proposal_without_setup_family")

        if proposal is not None:
            if proposal.forecast_id != forecast.forecast_id or proposal.market_state_id != state.state_id:
                errors.append("proposal_lineage_mismatch")
            proposal_references = set(proposal.supporting_evidence_ids) | set(proposal.contradicting_evidence_ids)
            if not proposal_references <= valid_evidence_ids:
                errors.append("proposal_unknown_evidence_reference")
            expected_key = structural_opportunity_key(
                state.instrument,
                forecast.selected_setup_family or "non_actionable",
                proposal.direction.value,
                self._structural_anchor_tokens(state, proposal.supporting_evidence_ids),
                forecast.dominant_scenario or "wait",
            )
            if proposal.structural_opportunity_key != expected_key:
                errors.append("structural_opportunity_key_mismatch")
            if proposal.created_at < request.analysis_timestamp:
                errors.append("proposal_timestamp_before_analysis")
            if proposal.created_at > forecast.generated_at + timedelta(minutes=1):
                errors.append("proposal_timestamp_after_forecast_window")
            if proposal.expires_at is not None and proposal.expires_at <= proposal.created_at:
                errors.append("proposal_expiry_invalid")
            existing = request.existing_signal_state
            if existing and proposal.stop_loss is not None and isinstance(existing.get("stop_loss"), (int, float)):
                old_stop = float(existing["stop_loss"])
                old_direction = str(existing.get("direction"))
                if (old_direction == "BUY" and proposal.stop_loss < old_stop) or (
                    old_direction == "SELL" and proposal.stop_loss > old_stop
                ):
                    errors.append("proposal_widens_existing_stop")
        if errors:
            raise StructuredAIOutputError(tuple(errors))
        return ValidatedAIOutput(forecast=forecast, proposal=proposal)

    @staticmethod
    def _available_evidence_kinds(state: UnifiedMarketState) -> set[str]:
        kinds = {
            item.source_engine
            for item in state.evidence
            if item.availability.value == "available"
        }
        raw_text = " ".join(str(item.raw_value).lower() for item in state.evidence if item.raw_value is not None)
        for token, kind in (
            ("order_block", "order_block"),
            ("fvg", "fvg"),
            ("fair_value_gap", "fvg"),
            ("displacement", "displacement"),
            ("session", "session"),
        ):
            if token in raw_text:
                kinds.add(kind)
        return kinds

    @staticmethod
    def _structural_anchor_tokens(state: UnifiedMarketState, evidence_ids: tuple[UUID, ...]) -> tuple[str, ...]:
        """Use stable structural object IDs when present, otherwise engine/timeframe identity.

        Phase 1 EvidenceItem IDs are intentionally state-specific, so using those IDs directly
        would create a new signal each candle. Structural objects and source roles persist.
        """
        selected = [item for item in state.evidence if item.evidence_id in set(evidence_ids)]
        tokens: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    lowered = key.lower()
                    if lowered in {"id", "zone_id", "pool_id", "order_block_id", "fvg_id", "sweep_id", "anchor_id"} and isinstance(child, (str, int)):
                        tokens.add(f"{lowered}:{child}")
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for item in selected:
            before = len(tokens)
            visit(item.raw_value)
            if len(tokens) == before:
                tokens.add(f"{item.source_engine}:{item.source_timeframe}")
        return tuple(sorted(tokens))
