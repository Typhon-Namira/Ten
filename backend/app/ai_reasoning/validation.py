"""Deterministic validation of untrusted LLM structured output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

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
    validation_issues: tuple[str, ...] = ()
    repaired_fields: tuple[str, ...] = ()

    @property
    def degraded_validation(self) -> bool:
        return bool(self.validation_issues or self.repaired_fields)


@dataclass(frozen=True)
class StructuredValidationIssue:
    field_path: str
    expected_type: str
    actual_value: Any
    validator_name: str
    offending_json_fragment: str
    recoverable: bool = False

    def encoded(self) -> str:
        return json.dumps(
            {
                "field_path": self.field_path,
                "expected_type": self.expected_type,
                "actual_value": self.actual_value,
                "validator_name": self.validator_name,
                "offending_json_fragment": self.offending_json_fragment,
                "recoverable": self.recoverable,
            },
            default=str,
            separators=(",", ":"),
        )


class StructuredAIOutputError(ValueError):
    def __init__(
        self,
        errors: tuple[str, ...],
        *,
        first_issue: StructuredValidationIssue | None = None,
    ) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
        self.first_issue = first_issue


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
        raw, repaired_fields, normalization_issues = self._normalize(raw, request=request, state=state, quant=quant)
        errors: list[str] = list(normalization_issues)
        if not isinstance(raw.get("forecast"), dict):
            issue = self._issue(
                ("forecast",),
                "object",
                raw.get("forecast"),
                "forecast_contract",
            )
            raise StructuredAIOutputError((issue.encoded(),), first_issue=issue)
        try:
            forecast = AIMarketForecast.model_validate(raw["forecast"])
        except ValidationError as exc:
            issues = tuple(self._pydantic_issue("forecast", item, raw["forecast"]) for item in exc.errors())
            raise StructuredAIOutputError(
                tuple(item.encoded() for item in issues),
                first_issue=issues[0] if issues else None,
            ) from exc
        proposal: AISignalProposal | None = None
        if raw.get("proposal") is not None:
            if not isinstance(raw["proposal"], dict):
                issue = self._issue(("proposal",), "object or null", raw["proposal"], "proposal_contract")
                errors.append(issue.encoded())
            else:
                try:
                    proposal = AISignalProposal.model_validate(raw["proposal"])
                except ValidationError as exc:
                    # A malformed proposal must not discard an otherwise useful forecast. The
                    # proposal is omitted and the persisted forecast is explicitly degraded.
                    errors.extend(
                        self._pydantic_issue("proposal", item, raw["proposal"]).encoded()
                        for item in exc.errors()
                    )

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
            domain_errors = tuple(item for item in errors if not item.startswith("{"))
            if domain_errors:
                raise StructuredAIOutputError(tuple(errors))
        return ValidatedAIOutput(
            forecast=forecast,
            proposal=proposal,
            validation_issues=tuple(errors),
            repaired_fields=repaired_fields,
        )

    def _normalize(
        self,
        raw: dict[str, Any],
        *,
        request: AIReasoningRequest,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
    ) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
        normalized = dict(raw)
        repaired: list[str] = []
        issues: list[str] = []
        if isinstance(normalized.get("forecast"), str) and str(normalized["forecast"]).strip().upper() == "WAIT":
            normalized = self._simplified_wait(request=request, state=state, quant=quant)
            repaired.extend(("forecast", "proposal"))
            issues.append(
                self._issue(
                    ("forecast",),
                    "AIMarketForecast object",
                    raw.get("forecast"),
                    "simplified_wait_normalizer",
                    recoverable=True,
                ).encoded()
            )

        forecast_raw = normalized.get("forecast")
        if isinstance(forecast_raw, dict):
            forecast = self._known_fields(dict(forecast_raw), AIMarketForecast, "forecast", repaired, issues)
            forecast.update(
                {
                    "forecast_id": forecast.get("forecast_id")
                    or str(uuid5(NAMESPACE_URL, f"ten:ai-forecast:{request.request_id}")),
                    "request_id": str(request.request_id),
                    "market_state_id": str(request.market_state_id),
                    "quantitative_forecast_id": str(request.quantitative_forecast_id),
                    "cycle_id": str(request.cycle_id),
                    "model_provider": "openrouter",
                    "model_identifier": request.model_identifier,
                    "prompt_version": request.prompt_version,
                    "reasoning_policy_version": request.reasoning_policy_version,
                    "setup_family_registry_version": request.setup_family_registry_version,
                    "quantitative_model_version": request.quantitative_model_version,
                    "feature_schema_version": request.feature_schema_version,
                    "market_state_schema_version": request.market_state_schema_version,
                    "validation_passed": False,
                    "retry_count": 0,
                    "shadow_only": True,
                    "awaiting_guardrail_validation": True,
                    "approved_for_publication": False,
                    "generated_at": request.created_at.isoformat(),
                }
            )
            self._normalize_enums(forecast, repaired)
            self._normalize_probabilities(forecast, repaired)
            normalized["forecast"] = forecast

        proposal_raw = normalized.get("proposal")
        if isinstance(proposal_raw, dict) and isinstance(normalized.get("forecast"), dict):
            proposal = self._known_fields(dict(proposal_raw), AISignalProposal, "proposal", repaired, issues)
            forecast = normalized["forecast"]
            proposal.update(
                {
                    "proposal_id": proposal.get("proposal_id")
                    or str(uuid5(NAMESPACE_URL, f"ten:ai-proposal:{request.request_id}")),
                    "forecast_id": forecast["forecast_id"],
                    "market_state_id": str(request.market_state_id),
                    "model_identifier": request.model_identifier,
                    "policy_version": request.reasoning_policy_version,
                    "shadow_only": True,
                    "awaiting_guardrail_validation": True,
                    "approved_for_publication": False,
                    "created_at": request.created_at.isoformat(),
                }
            )
            self._normalize_enums(proposal, repaired)
            valid_evidence = {str(item.evidence_id) for item in state.evidence}
            direction = str(proposal.get("direction", "NEUTRAL"))
            setup_family = str(forecast.get("selected_setup_family") or "non_actionable")
            scenario = str(forecast.get("dominant_scenario") or "wait")
            evidence_ids = tuple(
                UUID(str(value))
                for value in proposal.get("supporting_evidence_ids", ())
                if str(value) in valid_evidence
            )
            proposal["structural_opportunity_key"] = structural_opportunity_key(
                state.instrument,
                setup_family,
                direction,
                self._structural_anchor_tokens(state, evidence_ids),
                scenario,
            )
            normalized["proposal"] = proposal
        return normalized, tuple(dict.fromkeys(repaired)), tuple(issues)

    def _simplified_wait(
        self,
        *,
        request: AIReasoningRequest,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
    ) -> dict[str, Any]:
        probabilities = {
            key: float(value) if value is not None else 0.0
            for key, value in request.quantitative_probabilities.items()
        }
        total = sum(probabilities.values())
        if total <= 0:
            probabilities = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 1.0}
        else:
            probabilities = {key: value / total for key, value in probabilities.items()}
        direction = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
        forecast_id = uuid5(NAMESPACE_URL, f"ten:ai-forecast:{request.request_id}")
        forecast = {
            "forecast_id": str(forecast_id),
            "status": "non_actionable",
            "dominant_direction": direction,
            "buy_probability": probabilities.get("BUY", 0.0),
            "sell_probability": probabilities.get("SELL", 0.0),
            "neutral_probability": probabilities.get("NEUTRAL", 0.0),
            "expected_horizon": next(
                (item.horizon.horizon_id for item in quant.predictions),
                None,
            ),
            "dominant_scenario": "wait",
            "dominant_scenario_probability": probabilities[direction],
            "selected_setup_family": None,
            "setup_family_candidates": [],
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "missing_evidence": ["provider_returned_simplified_wait"],
            "evidence_completeness": state.evidence_completeness,
            "setup_readiness": "not_ready",
            "reasoning_summary": "Provider returned WAIT without the required analytical fields.",
            "fallback_state": "recovered_simplified_wait",
        }
        proposal = {
            "proposal_id": str(uuid5(NAMESPACE_URL, f"ten:ai-proposal:{request.request_id}")),
            "forecast_id": str(forecast_id),
            "market_state_id": str(request.market_state_id),
            "structural_opportunity_key": structural_opportunity_key(
                state.instrument,
                "non_actionable",
                "NEUTRAL",
                (),
                "wait",
            ),
            "recommended_action": "WAIT",
            "direction": "NEUTRAL",
            "setup_readiness": "not_ready",
            "proposal_confidence": 0.0,
            "reason_codes": ["provider_simplified_wait"],
        }
        return {"forecast": forecast, "proposal": proposal}

    @staticmethod
    def _known_fields(
        value: dict[str, Any],
        model: type[AIMarketForecast] | type[AISignalProposal],
        prefix: str,
        repaired: list[str],
        issues: list[str],
    ) -> dict[str, Any]:
        known = set(model.model_fields)
        unknown = sorted(set(value) - known)
        if unknown:
            repaired.extend(f"{prefix}.{field}" for field in unknown)
            issues.extend(
                StructuredAIOutputValidator._issue(
                    (prefix, field),
                    "known schema field",
                    value[field],
                    "unknown_field_filter",
                    recoverable=True,
                ).encoded()
                for field in unknown
            )
        return {key: item for key, item in value.items() if key in known}

    @staticmethod
    def _normalize_enums(value: dict[str, Any], repaired: list[str]) -> None:
        mappings = {
            "dominant_direction": {"BULLISH": "BUY", "BEARISH": "SELL", "LONG": "BUY", "SHORT": "SELL"},
            "direction": {"BULLISH": "BUY", "BEARISH": "SELL", "LONG": "BUY", "SHORT": "SELL"},
            "recommended_action": {"LONG": "BUY", "SHORT": "SELL", "NO_TRADE": "WAIT", "NONE": "WAIT"},
        }
        for field, aliases in mappings.items():
            if isinstance(value.get(field), str):
                original = value[field]
                upper = original.strip().upper()
                value[field] = aliases.get(upper, upper)
                if value[field] != original:
                    repaired.append(field)
        for field in ("status", "setup_readiness"):
            if isinstance(value.get(field), str):
                original = value[field]
                value[field] = original.strip().lower()
                if value[field] != original:
                    repaired.append(field)

    @staticmethod
    def _normalize_probabilities(value: dict[str, Any], repaired: list[str]) -> None:
        fields = ("buy_probability", "sell_probability", "neutral_probability")
        try:
            probabilities = [float(value[field]) for field in fields]
        except (KeyError, TypeError, ValueError):
            return
        total = sum(probabilities)
        if total > 0 and abs(total - 1.0) <= 0.05:
            for field, probability in zip(fields, probabilities, strict=True):
                value[field] = probability / total
            if abs(total - 1.0) > 1e-8:
                repaired.append("forecast.probabilities")

    @staticmethod
    def _issue(
        path: tuple[object, ...],
        expected: str,
        actual: Any,
        validator: str,
        *,
        recoverable: bool = False,
    ) -> StructuredValidationIssue:
        try:
            fragment = json.dumps(actual, default=str, separators=(",", ":"))[:500]
        except (TypeError, ValueError):
            fragment = str(actual)[:500]
        return StructuredValidationIssue(
            field_path=".".join(str(item) for item in path),
            expected_type=expected,
            actual_value=actual if isinstance(actual, (str, int, float, bool)) or actual is None else type(actual).__name__,
            validator_name=validator,
            offending_json_fragment=fragment,
            recoverable=recoverable,
        )

    @classmethod
    def _pydantic_issue(
        cls,
        prefix: str,
        error: Mapping[str, Any],
        raw: dict[str, Any],
    ) -> StructuredValidationIssue:
        path = tuple(error.get("loc", ()))
        actual: Any = raw
        for part in path:
            if isinstance(actual, dict):
                actual = actual.get(part)
            elif isinstance(actual, (list, tuple)) and isinstance(part, int) and part < len(actual):
                actual = actual[part]
            else:
                actual = None
                break
        expected = str(error.get("ctx", {}).get("expected") or error.get("msg") or error.get("type"))
        return cls._issue((prefix, *path), expected, actual, str(error.get("type", "pydantic")))

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
