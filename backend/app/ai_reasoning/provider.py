"""Ordered four-account Groq pool for TEN's AI reasoning boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
import logging
import random
import re
from time import perf_counter
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

from backend.app.ai.provider_client import (
    AIProviderClient,
    AIProviderCompletion,
    build_request_body,
    measure_request_body,
)
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.exceptions import AIProviderFailureDetails, AIProviderRequestError

from .analysis import AIAnalysisOutput
from .compact_output import (
    CompactAIAnalysisOutput,
    CompactOutputValidationError,
    CompactRetryAIAnalysisOutput,
    MARKET_REGIME_EVIDENCE_REF_LIMIT,
    normalize_descriptive_overflow,
    normalize_reference_syntax,
    resolve_compact_output,
    truncate_market_regime_evidence_refs,
    validate_evidence_references,
    validate_zone_references,
)
from .llm_context import build_llm_analysis_context, provider_context_payload
from .models import AIReasoningRequest
from .token_budget import OutputProfile, TokenBudgetManager

logger = logging.getLogger(__name__)
AI_REASONING_RESPONSE_SCHEMA_TYPE = "ten_ai_reasoning_response"
AI_REASONING_RESPONSE_SCHEMA_VERSION = "1.1"
_MAX_CORRECTION_FRAGMENT_CHARACTERS = 6_000


class ProviderStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DISABLED = "DISABLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ProviderRuntimeState:
    status: ProviderStatus
    model: str
    account_id: str
    enabled: bool
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    circuit_open_until: datetime | None = None
    last_failure_code: str | None = None
    last_http_status: int | None = None
    last_provider_error_code: str | None = None
    last_request_result: str | None = None
    last_attempt_schema_error: str | None = None
    last_success_schema_version: str | None = None
    request_policy_failures: int = 0
    recent_failures: list[datetime] = field(default_factory=list)
    calls_today: int = 0
    successful_analyses: int = 0
    provider_failures: int = 0
    rate_limit_failures: int = 0
    quota_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    metrics_date: str | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "account_id": self.account_id,
            "enabled": self.enabled,
            "configured": self.enabled,
            "eligible_now": self.status == ProviderStatus.AVAILABLE,
            "availability": self.status == ProviderStatus.AVAILABLE,
            "model": self.model,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "circuit_open_until": (
                self.circuit_open_until.isoformat() if self.circuit_open_until else None
            ),
            "last_failure_code": self.last_failure_code,
            "last_http_status": self.last_http_status,
            "last_provider_error_code": self.last_provider_error_code,
            "cooldown_until": (
                self.circuit_open_until.isoformat()
                if self.circuit_open_until
                else None
            ),
            "circuit_state": (
                "OPEN" if self.circuit_open_until is not None else "CLOSED"
            ),
            "rate_limit_state": (
                "ACTIVE" if self.status == ProviderStatus.RATE_LIMITED else "CLEAR"
            ),
            "quota_state": (
                "EXHAUSTED"
                if self.status == ProviderStatus.QUOTA_EXHAUSTED
                else "AVAILABLE"
            ),
            "last_request_status": self.last_http_status,
            "last_request_result": self.last_request_result,
            "latest_attempt_result": self.last_request_result,
            "latest_attempt_schema_error": self.last_attempt_schema_error,
            "latest_successful_attempt_at": (
                self.last_success_at.isoformat()
                if self.last_success_at
                else None
            ),
            "latest_success_schema_version": self.last_success_schema_version,
            "request_policy_health": (
                "degraded" if self.request_policy_failures else "healthy"
            ),
            "calls_today": self.calls_today,
            "successful_analyses": self.successful_analyses,
            "provider_failures": self.provider_failures,
            "rate_limit_failures": self.rate_limit_failures,
            "quota_failures": self.quota_failures,
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
        }


@dataclass(frozen=True)
class AIProviderResponse:
    raw_output: dict[str, Any]
    provider: str
    model_identifier: str
    latency_ms: float
    token_usage: dict[str, int] | None
    fallback_used: bool = False
    fallback_reason: str | None = None
    operational_metadata: dict[str, object] | None = None


class AIReasoningProvider(Protocol):
    async def reason(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
    ) -> AIProviderResponse: ...

    def metadata(self) -> dict[str, object]: ...


def reasoning_response_schema(
    profile: OutputProfile | str = OutputProfile.COMPACT,
) -> dict[str, Any]:
    """Strict application schema used for Groq JSON-output validation."""

    selected = OutputProfile(profile)
    model: type[BaseModel]
    if selected == OutputProfile.COMPACT_RETRY:
        model = CompactRetryAIAnalysisOutput
    elif selected == OutputProfile.COMPACT:
        model = CompactAIAnalysisOutput
    else:
        model = AIAnalysisOutput
    unsupported_keywords = {
        "title",
        "default",
        "description",
        "format",
        "minLength",
        "minItems",
        # Numeric bounds remain enforced by the unchanged Pydantic
        # domain schema after decoding. Omitting them from the wire
        # contract stays compact without weakening application validation.
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }

    def compact(value: Any, *, property_map: bool = False) -> Any:
        if isinstance(value, dict):
            return {
                key: compact(item, property_map=key == "properties")
                for key, item in value.items()
                # A model may legitimately define a property named
                # "description". Only remove schema metadata keywords, never
                # names inside a JSON Schema "properties" map.
                if property_map or key not in unsupported_keywords
                # Enum membership already constrains each value. The unchanged
                # application model enforces the concrete string type.
                if not (key == "type" and "enum" in value)
            }
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    return cast(dict[str, Any], compact(model.model_json_schema()))


def _schema_issue(exc: ValidationError) -> tuple[str, str, str, str]:
    errors = exc.errors()
    if not errors:
        return "schema_validation_failed", "provider_response", "schema validation", "{}"
    first = errors[0]
    location = ".".join(str(item) for item in first.get("loc", ()))
    path = ".".join(item for item in ("provider_response", location) if item)
    error_type = str(first.get("type") or "schema_validation")
    if path.endswith(("_supply_ref", "_demand_ref")) and error_type in {
        "float_type",
        "string_type",
        "string_pattern_mismatch",
    }:
        code = "wrong_reference_type"
    elif error_type == "missing":
        code = "missing_required_field"
    elif error_type == "extra_forbidden":
        code = "unexpected_field"
    elif error_type == "string_too_long":
        code = "text_too_long"
    elif error_type in {"too_long", "list_too_long", "tuple_too_long"}:
        code = "too_many_items"
    elif "enum" in error_type or "literal" in error_type:
        code = "invalid_enum"
    elif any(marker in error_type for marker in ("greater", "less", "multiple")):
        code = "numeric_range_error"
    elif "datetime" in error_type:
        code = "invalid_timestamp"
    elif error_type.startswith(("string_", "int_", "float_", "bool_", "tuple_", "list_")):
        code = "wrong_type"
    elif error_type in {"model_type", "model_attributes_type"}:
        code = "invalid_signal_shape"
    else:
        code = "business_rule_violation"
    fragment = json.dumps(
        {"path": path, "value": first.get("input")},
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )[:1000]
    return code, path, str(first.get("msg") or error_type), fragment


def _normalized_contract_output(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Remove only known response-contract metadata, never analytical values."""

    metadata = ("schema_type", "schema_version", "json_schema")
    removed = tuple(key for key in metadata if key in raw)
    if not removed:
        return raw, ()
    return {key: value for key, value in raw.items() if key not in metadata}, removed


def _combined_usage(*values: dict[str, int] | None) -> dict[str, int] | None:
    keys = ("input_tokens", "output_tokens", "total_tokens")
    combined = {
        key: sum(value.get(key, 0) for value in values if value and key in value)
        for key in keys
        if any(value and key in value for value in values)
    }
    return combined or None


def _required_object_fields(
    shape: object,
    path: str = "$",
) -> dict[str, tuple[str, ...]]:
    """Describe required object membership without repeating the full contract."""

    if not isinstance(shape, dict):
        return {}
    required = {path: tuple(shape)}
    for key, value in shape.items():
        required.update(_required_object_fields(value, f"{path}.{key}"))
    return required


class _OpenAICompatibleReasoningProvider:
    provider_name: str
    supports_strict_json_schema = True

    def __init__(
        self,
        client: AIProviderClient,
        prompts: PromptLoader,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        target_input_tokens: int,
        warning_input_tokens: int,
        hard_input_tokens: int,
        absolute_max_output_tokens: int,
        maximum_request_cost_usd: float,
        input_cost_per_million_usd: float,
        output_cost_per_million_usd: float,
        setup_family_ids: tuple[str, ...],
        output_profile: OutputProfile | str = OutputProfile.COMPACT,
        target_output_tokens: int | None = None,
        token_safety_margin: int = 256,
        model_context_limit: int = 8192,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.target_input_tokens = target_input_tokens
        self.warning_input_tokens = warning_input_tokens
        self.hard_input_tokens = hard_input_tokens
        self.absolute_max_output_tokens = absolute_max_output_tokens
        self.maximum_request_cost_usd = maximum_request_cost_usd
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd
        self.setup_family_ids = setup_family_ids
        self.token_budgets = TokenBudgetManager(
            model=model,
            output_profile=output_profile,
            model_context_limit=model_context_limit,
            maximum_input_tokens=min(target_input_tokens, hard_input_tokens),
            target_output_tokens=target_output_tokens,
            hard_output_limit=max_tokens,
            safety_margin_tokens=token_safety_margin,
        )
        self.http_calls = 0
        self.correction_attempts = 0

    @property
    def configured(self) -> bool:
        return self.client.configured and bool(self.model)

    async def reason(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
        attempt: int = 1,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        correction_instruction: str | None = None,
        previous_response_fragment: str | None = None,
        output_profile: OutputProfile | str | None = None,
        request_kind_override: str | None = None,
    ) -> AIProviderResponse:
        started = perf_counter()
        selected_profile = OutputProfile(
            output_profile or self.token_budgets.output_profile
        )
        context = build_llm_analysis_context(request)
        contract = self._response_contract(selected_profile, context)
        allowed_supply_refs: list[str | None] = [
            item.zone_id for item in context.supply_zone_catalog
        ] or [None]
        allowed_demand_refs: list[str | None] = [
            item.zone_id for item in context.demand_zone_catalog
        ] or [None]
        schema = reasoning_response_schema(selected_profile)
        included_sections: tuple[str, ...] = ()
        omitted_sections: tuple[str, ...] = ()
        if correction_instruction:
            system_prompt = (
                "Return exactly one compact JSON object and nothing else. "
                "Return the complete response contract with every required field; "
                "never return a partial patch or omit an unchanged object. "
                "Correct the supplied response according to validation_error. "
                "Do not invent market facts or emit trading actions."
            )
            previous_response: object = previous_response_fragment
            if previous_response_fragment:
                try:
                    # Keep a complete response as structured JSON. Passing an
                    # already-serialized JSON string makes the wire payload
                    # escape every quote and needlessly increases correction
                    # input tokens.
                    previous_response = json.loads(previous_response_fragment)
                except json.JSONDecodeError:
                    # A bounded malformed fragment is still useful for a JSON
                    # parse correction, but must remain explicitly a string.
                    previous_response = previous_response_fragment
            payload = {
                "validation_error": correction_instruction,
                "allowed_reference_values": {
                    "nearest_supply_ref": (
                        allowed_supply_refs
                    ),
                    "nearest_demand_ref": (
                        allowed_demand_refs
                    ),
                },
                "reference_rules": (
                    "Return one listed ID or null exactly; never return a price, "
                    "object, array, empty string, sentinel, or invented ID."
                ),
                "required_object_fields": _required_object_fields(
                    contract.get("shape")
                ),
                "response_contract_rules": contract.get("rules", ()),
                "complete_object_required": True,
                "previous_response": previous_response,
            }
        else:
            compact_context, included_sections, omitted_sections = (
                provider_context_payload(context, selected_profile)
            )
            payload = {
                "analysis_context": compact_context,
                "response_contract": contract,
            }
            system_prompt = self.prompts.load(prompt_version)
        plan = self.token_budgets.plan(
            system_prompt=system_prompt,
            context=(
                cast(dict[str, Any], payload.get("analysis_context"))
                if isinstance(payload.get("analysis_context"), dict)
                else payload
            ),
            schema=contract,
            profile=selected_profile,
            included_sections=included_sections,
            omitted_sections=omitted_sections,
        )
        wire_schema = schema if self.supports_strict_json_schema else None
        request_body = build_request_body(
            system_prompt=system_prompt,
            payload=payload,
            model=self.model,
            temperature=self.temperature,
            max_tokens=plan.hard_output_limit,
            response_schema=wire_schema,
        )
        metrics = measure_request_body(
            request_body,
            input_cost_per_million_usd=self.input_cost_per_million_usd,
            output_cost_per_million_usd=self.output_cost_per_million_usd,
        )
        logger.info(
            "ai_reasoning.request.measured",
            extra={
                "provider": self.provider_name,
                "request_id": str(request.request_id),
                "cycle_id": str(request.cycle_id),
                "model": self.model,
                "serialized_request_bytes": metrics.serialized_request_bytes,
                "estimated_input_tokens": metrics.estimated_input_tokens,
                "maximum_output_tokens": metrics.maximum_output_tokens,
                "estimated_maximum_cost_usd": metrics.estimated_maximum_cost_usd,
                "response_schema_bytes": metrics.response_schema_bytes,
                "output_profile": selected_profile.value,
                "target_output_tokens": plan.target_output_tokens,
                "hard_output_limit": plan.hard_output_limit,
                "token_estimator": plan.estimator,
                "input_budget_utilization_percent": (
                    plan.input_budget_utilization_percent
                ),
                "schema_token_cost": plan.schema_token_cost,
                "context_token_cost": plan.context_token_cost,
                "prompt_token_cost": plan.prompt_token_cost,
                "context_sections_included": plan.context_sections_included,
                "context_sections_omitted": plan.context_sections_omitted,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            },
        )
        rejection_reason: str | None = None
        if plan.hard_output_limit > self.absolute_max_output_tokens:
            rejection_reason = "maximum_output_tokens_exceeded"
        elif plan.estimated_input_tokens > plan.maximum_input_tokens:
            rejection_reason = "request_too_large"
        elif metrics.estimated_maximum_cost_usd > self.maximum_request_cost_usd:
            rejection_reason = "maximum_cost_exceeded"
        if rejection_reason is not None:
            raise AIProviderRequestError(
                AIProviderFailureDetails(
                    provider=self.provider_name,
                    reason_code=rejection_reason,
                    phase="request_validation",
                    endpoint=f"{self.client.base_url}/chat/completions",
                    model=self.model,
                    request_id=str(request.request_id),
                    cycle_id=str(request.cycle_id),
                    error_code=rejection_reason,
                    error_message=(
                        f"preflight rejected bytes={metrics.serialized_request_bytes} "
                        f"estimated_input_tokens={plan.estimated_input_tokens} "
                        f"maximum_input_tokens={plan.maximum_input_tokens} "
                        f"maximum_output_tokens={plan.hard_output_limit}"
                    ),
                    body_length=metrics.serialized_request_bytes,
                    exception_class="AIProviderRequestBudgetError",
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                )
            )
        # This increment is intentionally adjacent to the transport call. Preflight
        # rejections, scheduler ticks, cache reads, and dashboard reads never reach it.
        self.http_calls += 1
        request_kind = request_kind_override or (
            "schema_correction"
            if correction_instruction
            else "transport_retry"
            if attempt > 1
            else "analysis"
        )
        try:
            completion = await self.client.complete_json(
                system_prompt=system_prompt,
                payload=payload,
                model=self.model,
                temperature=self.temperature,
                max_tokens=plan.hard_output_limit,
                response_schema=wire_schema,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                instrument=request.instrument,
                ums_boundary=request.analysis_timestamp.isoformat(),
                trigger="five_minute_analysis_worker",
                idempotency_key=request.idempotency_key,
                time_bucket=(
                    request.analysis_time_bucket.isoformat()
                    if request.analysis_time_bucket
                    else None
                ),
                attempt=attempt,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                attempt_type=request_kind,
            )
        except AIProviderRequestError as exc:
            details = replace(
                exc.details,
                target_output_tokens=plan.target_output_tokens,
                hard_output_limit=plan.hard_output_limit,
                output_profile=plan.output_profile.value,
                analysis_schema_version=(
                    "compact-retry-1.1"
                    if plan.output_profile == OutputProfile.COMPACT_RETRY
                    else "compact-1.1"
                    if plan.output_profile == OutputProfile.COMPACT
                    else "standard-1.0"
                ),
                input_budget_utilization_percent=(
                    plan.input_budget_utilization_percent
                ),
                token_estimator=plan.estimator,
                context_sections_included=plan.context_sections_included,
                context_sections_omitted=plan.context_sections_omitted,
            )
            raise AIProviderRequestError(details) from exc
        return self._response(
            completion,
            started,
            fallback_used,
            fallback_reason,
            request_kind,
            attempt,
            plan,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider_name,
            "model_identifier": self.model,
            "configured": self.configured,
            "external_ai_apis": (self.provider_name,),
            "token_usage_available": True,
        }

    def _response(
        self,
        completion: AIProviderCompletion,
        started: float,
        fallback_used: bool,
        fallback_reason: str | None,
        request_kind: str,
        request_sequence: int,
        plan: Any,
    ) -> AIProviderResponse:
        decoded_json = json.dumps(
            completion.content,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw_json = completion.raw_json_text or decoded_json
        logger.info(
            "ai_provider.response.diagnostic",
            extra={
                "provider": self.provider_name,
                "model": self.model,
                "provider_request_id": completion.provider_request_id,
                "status_code": completion.status_code,
                "raw_response_sha256": sha256(raw_json.encode()).hexdigest(),
                "raw_response_character_count": len(raw_json),
                "decoded_response_sha256": sha256(decoded_json.encode()).hexdigest(),
                "decoded_response_character_count": len(decoded_json),
                "extraction_note": completion.extraction_note,
            },
        )
        return AIProviderResponse(
            raw_output=completion.content,
            provider=self.provider_name,
            model_identifier=self.model,
            latency_ms=(perf_counter() - started) * 1000,
            token_usage=completion.token_usage,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            operational_metadata={
                "provider_request_id": completion.provider_request_id,
                "status_code": completion.status_code,
                "rate_limit_limit": completion.rate_limit_limit,
                "rate_limit_remaining": completion.rate_limit_remaining,
                "rate_limit_reset": completion.rate_limit_reset,
                "rate_limit_request_limit": completion.rate_limit_request_limit,
                "rate_limit_request_remaining": completion.rate_limit_request_remaining,
                "rate_limit_request_reset": completion.rate_limit_request_reset,
                "rate_limit_token_limit": completion.rate_limit_token_limit,
                "rate_limit_token_remaining": completion.rate_limit_token_remaining,
                "rate_limit_token_reset": completion.rate_limit_token_reset,
                "retry_after": completion.retry_after,
                "finish_reason": completion.finish_reason,
                "response_size_bytes": completion.response_size_bytes,
                "response_character_count": completion.response_character_count,
                "prompt_character_count": completion.prompt_character_count,
                "extraction_note": completion.extraction_note,
                "request_kind": request_kind,
                "request_sequence": request_sequence,
                "target_output_tokens": plan.target_output_tokens,
                "hard_output_limit": plan.hard_output_limit,
                "output_profile": plan.output_profile.value,
                "analysis_schema_version": (
                    "compact-retry-1.1"
                    if plan.output_profile == OutputProfile.COMPACT_RETRY
                    else "compact-1.1"
                    if plan.output_profile == OutputProfile.COMPACT
                    else "standard-1.0"
                ),
                "input_budget_utilization_percent": (
                    plan.input_budget_utilization_percent
                ),
                "context_sections_included": plan.context_sections_included,
                "context_sections_omitted": plan.context_sections_omitted,
                "token_estimator": plan.estimator,
            },
        )

    def _response_contract(
        self,
        profile: OutputProfile = OutputProfile.COMPACT,
        context: Any | None = None,
    ) -> dict[str, Any]:
        supply_ids = (
            [item.zone_id for item in context.supply_zone_catalog]
            if context is not None
            else []
        )
        demand_ids = (
            [item.zone_id for item in context.demand_zone_catalog]
            if context is not None
            else []
        )
        supply_values: list[str | None] = supply_ids or [None]
        demand_values: list[str | None] = demand_ids or [None]
        rules = [
            "one JSON object; exact schema; no markdown or prose",
            "use only supplied evidence IDs",
            (
                "market_regime.evidence_refs must contain at most "
                f"{MARKET_REGIME_EVIDENCE_REF_LIMIT} catalog IDs ordered "
                "strongest to weakest"
            ),
            (
                "nearest_supply_ref must be exactly one valid supply ID"
                if supply_ids
                else "supply catalog empty; nearest_supply_ref must be null"
            ),
            (
                "nearest_demand_ref must be exactly one valid demand ID"
                if demand_ids
                else "demand catalog empty; nearest_demand_ref must be null"
            ),
            "reference fields contain IDs or null only; never prices or objects",
            "analysis only; no trade, proposal, execution, or private reasoning",
        ]
        if profile in {OutputProfile.STANDARD, OutputProfile.EXPANDED}:
            return {
                "json_schema": reasoning_response_schema(profile),
                "rules": rules,
            }
        if profile == OutputProfile.COMPACT_RETRY:
            shape = {
                "analysis_schema_version": "compact-retry-1.1",
                "output_profile": "compact_retry",
                "market_regime": {
                    "classification": "bullish|bearish|ranging|transitional|uncertain",
                    "strength": "0..100",
                    "confidence": "0..1",
                    "evidence_refs": (
                        f"array<={MARKET_REGIME_EVIDENCE_REF_LIMIT};"
                        "strongest-to-weakest"
                    ),
                },
                "higher_timeframe_context": {
                    "bias": "bullish|bearish|neutral|mixed|uncertain",
                    "summary": "text<=180",
                    "evidence_refs": "array<=2",
                },
                "market_structure": {
                    "short_term": "text<=120",
                    "medium_term": "text<=180",
                    "recent_change": "text<=120",
                    "evidence_refs": "array<=2",
                },
                "liquidity_analysis": {
                    "summary": "text<=180",
                    "events": "array<=2;text<=100",
                    "unresolved": "array<=2;text<=100",
                    "evidence_refs": "array<=2",
                },
                "supply_demand_analysis": {
                    "summary": "text<=160",
                    "nearest_supply_ref": supply_values,
                    "nearest_demand_ref": demand_values,
                    "evidence_refs": "array<=2",
                },
                "momentum_analysis": {
                    "direction": "bullish|bearish|neutral|mixed|uncertain",
                    "strength": "0..100",
                    "trend": "strengthening|weakening|stable|uncertain",
                    "evidence_refs": "array<=2",
                },
                "volatility_analysis": {
                    "state": "low|normal|high|extreme|uncertain",
                    "trend": "expanding|contracting|stable|uncertain",
                    "evidence_refs": "array<=2",
                },
                "bullish_evidence_refs": "array<=2",
                "bearish_evidence_refs": "array<=2",
                "contradiction_refs": "array<=2",
                "key_risk_refs": "array<=2",
                "invalidation_conditions": "array<=2;text<=160",
                "data_quality_warnings": "array<=2;text<=120",
                "analysis_confidence": "0..1",
            }
        else:
            retry_shape = self._response_contract(
                OutputProfile.COMPACT_RETRY,
                context,
            )["shape"]
            shape = {
                **cast(dict[str, Any], retry_shape),
                "analysis_schema_version": "compact-1.1",
                "output_profile": "compact",
                "bullish_evidence_refs": "array<=3",
                "bearish_evidence_refs": "array<=3",
                "contradiction_refs": "array<=3",
                "key_risk_refs": "array<=3",
                "data_quality_warnings": "array<=3;text<=120",
                "alternative_scenarios": (
                    "array<=2:{name<=60,description<=180,probability=0..1,"
                    "evidence_refs<=2}"
                ),
                "executive_summary": "text<=320",
            }
        return {
            "shape": shape,
            "rules": rules,
        }


class GroqProvider(_OpenAICompatibleReasoningProvider):
    provider_name = "groq"
    # JSON Object Mode plus TEN's unchanged application validator is the
    # portable contract for every account in the pool.
    supports_strict_json_schema = False

    def __init__(
        self,
        client: AIProviderClient,
        prompts: PromptLoader,
        *,
        account_id: str,
        model: str,
        temperature: float,
        max_tokens: int,
        target_input_tokens: int,
        warning_input_tokens: int,
        hard_input_tokens: int,
        absolute_max_output_tokens: int,
        maximum_request_cost_usd: float,
        input_cost_per_million_usd: float,
        output_cost_per_million_usd: float,
        setup_family_ids: tuple[str, ...],
        output_profile: OutputProfile | str = OutputProfile.COMPACT,
        target_output_tokens: int | None = None,
        token_safety_margin: int = 256,
        model_context_limit: int = 8192,
    ) -> None:
        self.provider_name = account_id
        super().__init__(
            client,
            prompts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            target_input_tokens=target_input_tokens,
            warning_input_tokens=warning_input_tokens,
            hard_input_tokens=hard_input_tokens,
            absolute_max_output_tokens=absolute_max_output_tokens,
            maximum_request_cost_usd=maximum_request_cost_usd,
            input_cost_per_million_usd=input_cost_per_million_usd,
            output_cost_per_million_usd=output_cost_per_million_usd,
            setup_family_ids=setup_family_ids,
            output_profile=output_profile,
            target_output_tokens=target_output_tokens,
            token_safety_margin=token_safety_margin,
            model_context_limit=model_context_limit,
        )
        self.request_attempts: list[dict[str, Any]] = []
        self.attempt_counters = {
            "analysis_requests": 0,
            "schema_correction_requests": 0,
            "http_429_responses": 0,
            "initial_parse_failures": 0,
            "initial_schema_validation_failures": 0,
            "schema_corrections_succeeded": 0,
            "schema_corrections_failed": 0,
            "truncated_outputs": 0,
            "compact_retries": 0,
            "request_policy_failures": 0,
            "provider_http_successes": 0,
            "schema_valid_analyses": 0,
            "provider_input_tokens": 0,
            "provider_output_tokens": 0,
            "provider_total_tokens": 0,
        }

    def _record_attempt(
        self,
        request: AIReasoningRequest,
        *,
        request_kind: str,
        request_sequence: int,
        response: AIProviderResponse | None = None,
        error: AIProviderFailureDetails | None = None,
        schema_valid: bool | None = None,
        schema_error_code: str | None = None,
        schema_error_path: str | None = None,
        correction_triggered: bool = False,
    ) -> None:
        metadata = (response.operational_metadata or {}) if response else {}
        usage = response.token_usage if response else None
        context = build_llm_analysis_context(request)
        supply_demand = (
            response.raw_output.get("supply_demand_analysis")
            if response
            else None
        )
        reference_key = (
            "nearest_supply_ref"
            if schema_error_path and schema_error_path.endswith("nearest_supply_ref")
            else "nearest_demand_ref"
            if schema_error_path and schema_error_path.endswith("nearest_demand_ref")
            else None
        )
        received_reference = (
            supply_demand.get(reference_key)
            if isinstance(supply_demand, dict) and reference_key
            else None
        )
        attempt_seed = (
            f"{request.request_id}:{self.provider_name}:"
            f"{request_sequence}:{request_kind}"
        )
        item: dict[str, Any] = {
            "analysis_job_id": str(request.request_id),
            "eligible_cycle_id": str(request.cycle_id),
            "provider_attempt_id": sha256(attempt_seed.encode()).hexdigest()[:24],
            "account_id": self.provider_name,
            "request_kind": request_kind,
            "request_sequence": request_sequence,
            "recorded_at": datetime.now(UTC).isoformat(),
            "input_tokens": (
                usage.get("input_tokens")
                if usage
                else error.provider_input_tokens if error else None
            ),
            "output_tokens": (
                usage.get("output_tokens")
                if usage
                else error.provider_output_tokens if error else None
            ),
            "total_tokens": (
                usage.get("total_tokens")
                if usage
                else error.provider_total_tokens if error else None
            ),
            "finish_reason": (
                metadata.get("finish_reason") if response else error.finish_reason
                if error
                else None
            ),
            "http_status": (
                metadata.get("status_code") if response else error.http_status
                if error
                else None
            ),
            "provider_error_code": (
                error.error_code if error else None
            ),
            "provider_request_id": (
                metadata.get("provider_request_id")
                if response
                else error.provider_request_id if error else None
            ),
            "latency_ms": (
                response.latency_ms if response else error.elapsed_ms if error else None
            ),
            "response_size_bytes": (
                metadata.get("response_size_bytes")
                if response
                else error.response_size_bytes if error else None
            ),
            "response_character_count": (
                metadata.get("response_character_count")
                if response
                else error.response_character_count if error else None
            ),
            "prompt_character_count": (
                metadata.get("prompt_character_count") if response else None
            ),
            "target_output_tokens": (
                metadata.get("target_output_tokens")
                if response
                else error.target_output_tokens if error else None
            ),
            "hard_output_limit": (
                metadata.get("hard_output_limit")
                if response
                else error.hard_output_limit if error else None
            ),
            "output_profile": (
                metadata.get("output_profile")
                if response
                else error.output_profile if error else None
            ),
            "analysis_schema_version": (
                metadata.get("analysis_schema_version")
                if response
                else error.analysis_schema_version if error else None
            ),
            "input_budget_utilization_percent": (
                metadata.get("input_budget_utilization_percent")
                if response
                else error.input_budget_utilization_percent if error else None
            ),
            "output_budget_utilization_percent": (
                round(
                    usage["output_tokens"]
                    / max(1, cast(int, metadata["hard_output_limit"]))
                    * 100,
                    2,
                )
                if usage
                and isinstance(usage.get("output_tokens"), int)
                and isinstance(metadata.get("hard_output_limit"), int)
                else None
            ),
            "token_estimator": (
                metadata.get("token_estimator")
                if response
                else error.token_estimator if error else None
            ),
            "context_sections_included": (
                metadata.get("context_sections_included", ())
                if response
                else error.context_sections_included if error else ()
            ),
            "context_sections_omitted": (
                metadata.get("context_sections_omitted", ())
                if response
                else error.context_sections_omitted if error else ()
            ),
            "fields_completed_before_failure": (
                len(response.raw_output) if response else None
            ),
            "schema_valid": schema_valid,
            "schema_error_code": schema_error_code or (
                error.schema_error_code if error else None
            ),
            "schema_error_path": schema_error_path or (
                error.schema_error_path if error else None
            ),
            "schema_correction_triggered": correction_triggered,
            "compact_retry_triggered": request_kind == "compact_retry",
            "limit_classification": (
                error.limit_classification if error else None
            ),
            "supply_catalog_count": len(context.supply_zone_catalog),
            "demand_catalog_count": len(context.demand_zone_catalog),
            "liquidity_catalog_count": len(context.nearest_liquidity_levels),
            "evidence_catalog_count": len(context.evidence_catalog),
            "valid_supply_refs": tuple(
                item.zone_id for item in context.supply_zone_catalog
            ),
            "valid_demand_refs": tuple(
                item.zone_id for item in context.demand_zone_catalog
            ),
            "selected_supply_ref": (
                supply_demand.get("nearest_supply_ref")
                if isinstance(supply_demand, dict)
                and isinstance(
                    supply_demand.get("nearest_supply_ref"),
                    str,
                )
                else None
            ),
            "selected_demand_ref": (
                supply_demand.get("nearest_demand_ref")
                if isinstance(supply_demand, dict)
                and isinstance(
                    supply_demand.get("nearest_demand_ref"),
                    str,
                )
                else None
            ),
            "reference_validation_result": (
                "valid"
                if schema_valid
                else schema_error_code
                if reference_key
                else None
            ),
            "received_reference_type": (
                type(received_reference).__name__
                if reference_key
                else None
            ),
            "received_reference_value_hash": (
                sha256(
                    json.dumps(
                        received_reference,
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest()[:16]
                if reference_key
                else None
            ),
        }
        self.request_attempts.append(item)
        del self.request_attempts[:-100]
        counter = (
            "schema_correction_requests"
            if request_kind == "schema_correction"
            else "analysis_requests"
            if request_kind == "analysis"
            else None
        )
        if counter:
            self.attempt_counters[counter] += 1
        if item["http_status"] == 429:
            self.attempt_counters["http_429_responses"] += 1
        if item["http_status"] == 200:
            self.attempt_counters["provider_http_successes"] += 1
        for usage_key, counter_key in (
            ("input_tokens", "provider_input_tokens"),
            ("output_tokens", "provider_output_tokens"),
            ("total_tokens", "provider_total_tokens"),
        ):
            usage_value = item.get(usage_key)
            if isinstance(usage_value, int):
                self.attempt_counters[counter_key] += usage_value
        if schema_valid:
            self.attempt_counters["schema_valid_analyses"] += 1
        if schema_error_code in {"finish_reason_length", "truncated_response"}:
            self.attempt_counters["truncated_outputs"] += 1
        if request_kind == "compact_retry":
            self.attempt_counters["compact_retries"] += 1
        if request_kind == "analysis" and schema_valid is False:
            if schema_error_code in {
                "empty_response",
                "truncated_response",
                "finish_reason_length",
                "json_not_found",
                "json_parse_error",
                "multiple_json_objects",
                "response_envelope_invalid",
            }:
                self.attempt_counters["initial_parse_failures"] += 1
            else:
                self.attempt_counters[
                    "initial_schema_validation_failures"
                ] += 1
        if request_kind == "schema_correction":
            self.attempt_counters[
                "schema_corrections_succeeded"
                if schema_valid
                else "schema_corrections_failed"
            ] += 1
        event = (
            "groq.request.rate_limited"
            if item["http_status"] == 429
            else "groq.request.completed"
        )
        # Railway's default log collector retains the formatted message but
        # drops Python LogRecord ``extra`` fields. Append only this explicitly
        # allow-listed diagnostic object so production logs remain useful
        # without exposing prompts, responses, or credentials.
        logger.info(
            "%s %s",
            event,
            json.dumps(item, default=str, separators=(",", ":")),
            extra=item,
        )

    def attempts_for(self, request_id: object) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(item)
            for item in self.request_attempts
            if item["analysis_job_id"] == str(request_id)
        )

    async def reason(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
        attempt: int = 1,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        correction_instruction: str | None = None,
        previous_response_fragment: str | None = None,
        output_profile: OutputProfile | str | None = None,
        request_kind_override: str | None = None,
    ) -> AIProviderResponse:
        del previous_response_fragment, output_profile, request_kind_override
        request_kind = "schema_correction" if correction_instruction else "analysis"
        initial_response: AIProviderResponse | None = None
        previous_fragment: str | None = None
        try:
            response = await super().reason(
                request,
                prompt_version=prompt_version,
                attempt=attempt,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                correction_instruction=correction_instruction,
                output_profile=OutputProfile.COMPACT,
            )
        except AIProviderRequestError as exc:
            self._record_attempt(
                request,
                request_kind=request_kind,
                request_sequence=attempt,
                error=exc.details,
                schema_valid=False,
                schema_error_code=exc.details.schema_error_code,
            )
            if exc.details.reason_code == "truncated_response":
                return await self._compact_retry(
                    request,
                    prompt_version=prompt_version,
                    attempt=attempt + 1,
                    fallback_used=fallback_used,
                    fallback_reason="output_truncated",
                    initial_usage=self._failure_usage(exc.details),
                )
            if exc.details.reason_code not in {"response_decoding_failed"}:
                raise
            correction_instruction = (
                "The previous response was not valid JSON. Return valid JSON matching "
                "the complete response contract."
            )
            correction_reason = exc.details.schema_error_code or exc.details.reason_code
        else:
            normalized, removed = _normalized_contract_output(response.raw_output)
            if removed:
                response = replace(response, raw_output=normalized)
            finish_reason = (
                response.operational_metadata.get("finish_reason")
                if response.operational_metadata
                else None
            )
            if finish_reason == "length":
                self._record_attempt(
                    request,
                    request_kind=request_kind,
                    request_sequence=attempt,
                    response=response,
                    schema_valid=False,
                    schema_error_code="finish_reason_length",
                    schema_error_path="provider_response",
                )
                return await self._compact_retry(
                    request,
                    prompt_version=prompt_version,
                    attempt=attempt + 1,
                    fallback_used=fallback_used,
                    fallback_reason="output_truncated",
                    initial_usage=response.token_usage,
                )
            try:
                resolved = self._validate_and_resolve(
                    response,
                    request,
                    OutputProfile.COMPACT,
                )
                self._record_attempt(
                    request,
                    request_kind=request_kind,
                    request_sequence=attempt,
                    response=resolved,
                    schema_valid=True,
                )
                return resolved
            except ValidationError as exc:
                code, field_path, expected, fragment = _schema_issue(exc)
            except CompactOutputValidationError as exc:
                code = exc.code
                field_path = exc.path
                expected = str(exc)
                fragment = "{}"
            if "code" in locals():
                correction_instruction = (
                    f"{code} at {field_path}: {expected}. "
                    f"Invalid fragment: {fragment}"
                )
                correction_reason = code
                previous_fragment = json.dumps(
                    response.raw_output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )[:_MAX_CORRECTION_FRAGMENT_CHARACTERS]
                initial_response = response
                metadata = response.operational_metadata or {}
                correction_capacity_exhausted = any(
                    str(metadata.get(name, "")).strip() == "0"
                    for name in (
                        "rate_limit_request_remaining",
                        "rate_limit_token_remaining",
                    )
                )
                self._record_attempt(
                    request,
                    request_kind=request_kind,
                    request_sequence=attempt,
                    response=response,
                    schema_valid=False,
                    schema_error_code=code,
                    schema_error_path=field_path,
                    correction_triggered=not correction_capacity_exhausted,
                )
                if correction_capacity_exhausted:
                    raise AIProviderRequestError(
                        AIProviderFailureDetails(
                            provider=self.provider_name,
                            reason_code="rate_limited",
                            phase="schema_correction_capacity",
                            endpoint=f"{self.client.base_url}/chat/completions",
                            model=self.model,
                            request_id=str(request.request_id),
                            cycle_id=str(request.cycle_id),
                            http_status=429,
                            error_code="correction_capacity_exhausted",
                            limit_classification="RATE_LIMITED_TEMPORARY",
                            schema_error_code=code,
                            schema_error_path=field_path,
                            rate_limit_request_remaining=cast(
                                str | None,
                                metadata.get("rate_limit_request_remaining"),
                            ),
                            rate_limit_token_remaining=cast(
                                str | None,
                                metadata.get("rate_limit_token_remaining"),
                            ),
                            rate_limit_request_reset=cast(
                                str | None,
                                metadata.get("rate_limit_request_reset"),
                            ),
                            rate_limit_token_reset=cast(
                                str | None,
                                metadata.get("rate_limit_token_reset"),
                            ),
                            exception_class="SchemaCorrectionCapacityUnavailable",
                        )
                    )

        logger.warning(
            "ai_provider.correction.started",
            extra={
                "provider": self.provider_name,
                "model": self.model,
                "instrument": request.instrument,
                "cycle_id": str(request.cycle_id),
                "ums_boundary": request.analysis_timestamp.isoformat(),
                "attempt": attempt + 1,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "sanitized_error_code": correction_reason,
                "analysis_job_id": str(request.request_id),
                "eligible_cycle_id": str(request.cycle_id),
                "account_id": self.provider_name,
                "request_kind": "schema_correction",
                "request_sequence": attempt + 1,
            },
        )
        # One and only one bounded correction request. A second malformed or
        # schema-invalid response is returned to the application validator, which
        # persists the typed failure and keeps publication fail-closed.
        self.correction_attempts += 1
        try:
            corrected = await super().reason(
                request,
                prompt_version=prompt_version,
                attempt=attempt + 1,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                correction_instruction=correction_instruction,
                previous_response_fragment=previous_fragment,
                output_profile=OutputProfile.COMPACT,
            )
        except AIProviderRequestError as exc:
            self._record_attempt(
                request,
                request_kind="schema_correction",
                request_sequence=attempt + 1,
                error=exc.details,
                schema_valid=False,
                schema_error_code=correction_reason,
            )
            raise AIProviderRequestError(
                replace(
                    exc.details,
                    phase=f"schema_correction_{exc.details.phase}",
                    schema_error_code=(
                        exc.details.schema_error_code or correction_reason
                    ),
                )
            ) from exc
        corrected_raw, _ = _normalized_contract_output(corrected.raw_output)
        corrected = replace(corrected, raw_output=corrected_raw)
        try:
            resolved = self._validate_and_resolve(
                corrected,
                request,
                OutputProfile.COMPACT,
            )
        except ValidationError as exc:
            code, path, _, _ = _schema_issue(exc)
            self._record_attempt(
                request,
                request_kind="schema_correction",
                request_sequence=attempt + 1,
                response=corrected,
                schema_valid=False,
                schema_error_code=code,
                schema_error_path=path,
            )
            raise self._schema_validation_error(
                request,
                corrected,
                code=code,
                path=path,
            ) from exc
        except CompactOutputValidationError as exc:
            self._record_attempt(
                request,
                request_kind="schema_correction",
                request_sequence=attempt + 1,
                response=corrected,
                schema_valid=False,
                schema_error_code=exc.code,
                schema_error_path=exc.path,
            )
            raise self._schema_validation_error(
                request,
                corrected,
                code=exc.code,
                path=exc.path,
            ) from exc
        else:
            self._record_attempt(
                request,
                request_kind="schema_correction",
                request_sequence=attempt + 1,
                response=resolved,
                schema_valid=True,
            )
        return replace(
            resolved,
            token_usage=_combined_usage(
                initial_response.token_usage if initial_response else None,
                corrected.token_usage,
            ),
        )

    @staticmethod
    def _failure_usage(
        details: AIProviderFailureDetails,
    ) -> dict[str, int] | None:
        usage = {
            key: value
            for key, value in (
                ("input_tokens", details.provider_input_tokens),
                ("output_tokens", details.provider_output_tokens),
                ("total_tokens", details.provider_total_tokens),
            )
            if value is not None
        }
        return usage or None

    def _validate_and_resolve(
        self,
        response: AIProviderResponse,
        request: AIReasoningRequest,
        profile: OutputProfile,
    ) -> AIProviderResponse:
        context = build_llm_analysis_context(request)
        normalized, evidence_ref_truncations = (
            truncate_market_regime_evidence_refs(
                response.raw_output,
                frozenset(
                    item.evidence_id for item in context.evidence_catalog
                ),
            )
        )
        normalized, reference_changes = normalize_reference_syntax(normalized)
        normalized, descriptive_changes = normalize_descriptive_overflow(
            normalized
        )
        if "analysis_schema_version" not in normalized:
            standard = AIAnalysisOutput.model_validate(normalized)
            metadata = dict(response.operational_metadata or {})
            metadata["output_profile"] = "standard"
            metadata["analysis_schema_version"] = "standard-1.0"
            metadata["local_descriptive_normalizations"] = descriptive_changes
            metadata["local_reference_normalizations"] = reference_changes
            return replace(
                response,
                raw_output=standard.model_dump(mode="json"),
                operational_metadata=metadata,
            )
        wire: CompactAIAnalysisOutput | CompactRetryAIAnalysisOutput
        if profile == OutputProfile.COMPACT_RETRY:
            wire = CompactRetryAIAnalysisOutput.model_validate(normalized)
        else:
            wire = CompactAIAnalysisOutput.model_validate(normalized)
        validate_evidence_references(wire, context.evidence_catalog)
        validate_zone_references(
            wire,
            context.supply_zone_catalog,
            context.demand_zone_catalog,
        )
        resolved = resolve_compact_output(
            wire,
            context.evidence_catalog,
            context.supply_zone_catalog,
            context.demand_zone_catalog,
        )
        metadata = dict(response.operational_metadata or {})
        metadata["local_descriptive_normalizations"] = descriptive_changes
        metadata["local_reference_normalizations"] = reference_changes
        metadata["local_evidence_ref_truncations"] = evidence_ref_truncations
        metadata["supply_catalog_count"] = len(context.supply_zone_catalog)
        metadata["demand_catalog_count"] = len(context.demand_zone_catalog)
        metadata["evidence_catalog_count"] = len(context.evidence_catalog)
        metadata["valid_supply_refs"] = tuple(
            item.zone_id for item in context.supply_zone_catalog
        )
        metadata["valid_demand_refs"] = tuple(
            item.zone_id for item in context.demand_zone_catalog
        )
        metadata["selected_supply_ref"] = (
            wire.supply_demand_analysis.nearest_supply_ref
        )
        metadata["selected_demand_ref"] = (
            wire.supply_demand_analysis.nearest_demand_ref
        )
        metadata["reference_validation_result"] = "valid"
        return replace(
            response,
            raw_output=resolved.model_dump(mode="json"),
            operational_metadata=metadata,
        )

    async def _compact_retry(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
        attempt: int,
        fallback_used: bool,
        fallback_reason: str,
        initial_usage: dict[str, int] | None,
    ) -> AIProviderResponse:
        logger.warning(
            "ai_provider.compact_retry.started",
            extra={
                "provider": self.provider_name,
                "analysis_job_id": str(request.request_id),
                "eligible_cycle_id": str(request.cycle_id),
                "request_kind": "compact_retry",
                "request_sequence": attempt,
                "previous_output_included": False,
            },
        )
        try:
            retry = await super().reason(
                request,
                prompt_version=prompt_version,
                attempt=attempt,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                output_profile=OutputProfile.COMPACT_RETRY,
                request_kind_override="compact_retry",
            )
        except AIProviderRequestError as exc:
            self._record_attempt(
                request,
                request_kind="compact_retry",
                request_sequence=attempt,
                error=exc.details,
                schema_valid=False,
                schema_error_code=(
                    "finish_reason_length"
                    if exc.details.reason_code == "truncated_response"
                    else exc.details.schema_error_code
                ),
            )
            if exc.details.reason_code != "truncated_response":
                raise
            self.attempt_counters["request_policy_failures"] += 1
            raise self._output_budget_error(request, exc.details) from exc
        finish_reason = (
            retry.operational_metadata.get("finish_reason")
            if retry.operational_metadata
            else None
        )
        if finish_reason == "length":
            self._record_attempt(
                request,
                request_kind="compact_retry",
                request_sequence=attempt,
                response=retry,
                schema_valid=False,
                schema_error_code="finish_reason_length",
            )
            self.attempt_counters["request_policy_failures"] += 1
            raise self._output_budget_error(request)
        try:
            resolved = self._validate_and_resolve(
                retry,
                request,
                OutputProfile.COMPACT_RETRY,
            )
        except (ValidationError, CompactOutputValidationError) as exc:
            if isinstance(exc, ValidationError):
                code, path, _, _ = _schema_issue(exc)
            else:
                code, path = exc.code, exc.path
            self._record_attempt(
                request,
                request_kind="compact_retry",
                request_sequence=attempt,
                response=retry,
                schema_valid=False,
                schema_error_code=code,
                schema_error_path=path,
            )
            raise AIProviderRequestError(
                AIProviderFailureDetails(
                    provider=self.provider_name,
                    reason_code="schema_validation_error",
                    phase="compact_retry_validation",
                    endpoint=f"{self.client.base_url}/chat/completions",
                    model=self.model,
                    request_id=str(request.request_id),
                    cycle_id=str(request.cycle_id),
                    http_status=200,
                    schema_error_code=code,
                    schema_error_path=path,
                    exception_class=type(exc).__name__,
                )
            ) from exc
        self._record_attempt(
            request,
            request_kind="compact_retry",
            request_sequence=attempt,
            response=resolved,
            schema_valid=True,
        )
        return replace(
            resolved,
            token_usage=_combined_usage(initial_usage, retry.token_usage),
            fallback_used=True,
            fallback_reason="output_truncated_compact_retry",
        )

    def _output_budget_error(
        self,
        request: AIReasoningRequest,
        details: AIProviderFailureDetails | None = None,
    ) -> AIProviderRequestError:
        return AIProviderRequestError(
            AIProviderFailureDetails(
                provider=self.provider_name,
                reason_code="output_budget_exceeded",
                phase="output_budget_policy",
                endpoint=f"{self.client.base_url}/chat/completions",
                model=self.model,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                http_status=200,
                finish_reason="length",
                schema_error_code="OUTPUT_BUDGET_EXCEEDED",
                provider_input_tokens=(
                    details.provider_input_tokens if details else None
                ),
                provider_output_tokens=(
                    details.provider_output_tokens if details else None
                ),
                provider_total_tokens=(
                    details.provider_total_tokens if details else None
                ),
                target_output_tokens=(
                    details.target_output_tokens if details else None
                ),
                hard_output_limit=details.hard_output_limit if details else None,
                output_profile=details.output_profile if details else None,
                analysis_schema_version=(
                    details.analysis_schema_version if details else None
                ),
                input_budget_utilization_percent=(
                    details.input_budget_utilization_percent
                    if details
                    else None
                ),
                token_estimator=details.token_estimator if details else None,
                context_sections_included=(
                    details.context_sections_included if details else ()
                ),
                context_sections_omitted=(
                    details.context_sections_omitted if details else ()
                ),
                exception_class="OutputBudgetExceeded",
            )
        )

    def _schema_validation_error(
        self,
        request: AIReasoningRequest,
        response: AIProviderResponse,
        *,
        code: str,
        path: str,
    ) -> AIProviderRequestError:
        usage = response.token_usage or {}
        metadata = response.operational_metadata or {}
        return AIProviderRequestError(
            AIProviderFailureDetails(
                provider=self.provider_name,
                reason_code="schema_validation_error",
                phase="structured_output_validation",
                endpoint=f"{self.client.base_url}/chat/completions",
                model=self.model,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                http_status=200,
                finish_reason=cast(str | None, metadata.get("finish_reason")),
                schema_error_code=code,
                schema_error_path=path,
                provider_input_tokens=usage.get("input_tokens"),
                provider_output_tokens=usage.get("output_tokens"),
                provider_total_tokens=usage.get("total_tokens"),
                output_profile=cast(
                    str | None,
                    metadata.get("output_profile"),
                ),
                analysis_schema_version=cast(
                    str | None,
                    metadata.get("analysis_schema_version"),
                ),
                exception_class="SchemaValidationError",
            )
        )


class GroqProviderPool:
    """Ordered Groq account failover with independent account cooldowns."""

    def __init__(
        self,
        providers: tuple[GroqProvider, ...],
        *,
        maximum_retries: int = 1,
        rate_limit_cooldown_seconds: float = 3600,
        quota_cooldown_seconds: float = 86400,
        configuration_cooldown_seconds: float = 86400,
        transport_circuit_seconds: float = 300,
        circuit_failure_threshold: int = 2,
        circuit_rolling_window_seconds: float = 60,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if len(providers) != 4:
            raise ValueError("GroqProviderPool requires exactly four account slots")
        self.providers = providers
        self.providers_by_id = {
            provider.provider_name: provider for provider in providers
        }
        self.maximum_retries = maximum_retries
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self.quota_cooldown_seconds = quota_cooldown_seconds
        self.configuration_cooldown_seconds = configuration_cooldown_seconds
        self.transport_circuit_seconds = transport_circuit_seconds
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_rolling_window_seconds = circuit_rolling_window_seconds
        self.clock = clock or (lambda: datetime.now(UTC))
        self.states = {
            provider.provider_name: ProviderRuntimeState(
                status=(
                    ProviderStatus.AVAILABLE
                    if provider.configured
                    else ProviderStatus.DISABLED
                ),
                model=provider.model,
                account_id=provider.provider_name,
                enabled=provider.configured,
            )
            for provider in providers
        }
        self.active_provider: str | None = None
        self.latest_successful_analysis_at: datetime | None = None
        self.retry_attempts = 0
        # Monotonic process counters are separate from the UTC-day runtime
        # counters. Service-level metric deltas are persisted per request, so
        # they must not roll backwards at midnight.
        self._telemetry = {
            provider.provider_name: {
                "calls": 0,
                "provider_failures": 0,
                "rate_limit_failures": 0,
                "quota_failures": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            for provider in providers
        }

    async def reason(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
    ) -> AIProviderResponse:
        now = self.clock()
        last_error: AIProviderRequestError | None = None
        logger.info(
            "groq_pool.routing.entered",
            extra={
                "provider": "groq_pool",
                "instrument": request.instrument,
                "cycle_id": str(request.cycle_id),
                "ums_boundary": request.analysis_timestamp.isoformat(),
                "strategy": "ordered_failover",
                "account_order": tuple(self.providers_by_id),
            },
        )
        for index, provider in enumerate(self.providers):
            account_id = provider.provider_name
            if not self._eligible(account_id, now):
                account_state = self.states[account_id]
                cooldown_until = account_state.circuit_open_until
                logger.info(
                    "groq_pool.account.skipped",
                    extra={
                        "provider": account_id,
                        "model": provider.model,
                        "cycle_id": str(request.cycle_id),
                        "status": account_state.status.value,
                        "cooldown_until": (
                            cooldown_until.isoformat()
                            if cooldown_until
                            else None
                        ),
                    },
                )
                continue
            logger.info(
                "groq_pool.account.started",
                extra={
                    "provider": account_id,
                    "model": provider.model,
                    "instrument": request.instrument,
                    "cycle_id": str(request.cycle_id),
                    "ums_boundary": request.analysis_timestamp.isoformat(),
                    "account_position": index + 1,
                },
            )
            try:
                response = await self._attempt(
                    provider,
                    request,
                    prompt_version,
                    fallback_used=index > 0,
                    fallback_reason=(
                        last_error.details.reason_code if last_error else None
                    ),
                )
                self._success(account_id, response)
                return response
            except AIProviderRequestError as exc:
                last_error = exc
                if exc.details.reason_code in {
                    "output_budget_exceeded",
                    "schema_validation_error",
                }:
                    self._policy_failure(account_id, exc.details)
                else:
                    self._failure(account_id, exc.details)
                logger.warning(
                    "groq_pool.account.failed",
                    extra={
                        "provider": account_id,
                        "model": provider.model,
                        "instrument": request.instrument,
                        "cycle_id": str(request.cycle_id),
                        "ums_boundary": request.analysis_timestamp.isoformat(),
                        "status_code": exc.details.http_status,
                        "sanitized_error_code": exc.details.reason_code,
                        "next_account_allowed": self._failover_allowed(exc.details),
                    },
                )
                if not self._failover_allowed(exc.details):
                    raise
        pool_metadata = self.metadata()
        unavailable_diagnostic = {
            "analysis_job_id": str(request.request_id),
            "eligible_cycle_id": str(request.cycle_id),
            "instrument": request.instrument,
            "configured_account_count": pool_metadata["configured_account_count"],
            "available_account_count": pool_metadata["available_account_count"],
            "temporary_rate_limited_account_count": pool_metadata[
                "temporary_rate_limited_account_count"
            ],
            "quota_exhausted_account_count": pool_metadata[
                "quota_exhausted_account_count"
            ],
            "configuration_error_account_count": pool_metadata[
                "configuration_error_account_count"
            ],
            "aggregate_reason": pool_metadata["aggregate_reason"],
            "last_failure_code": (
                last_error.details.reason_code if last_error else None
            ),
            "next_retry_at": min(
                (
                    state.circuit_open_until.isoformat()
                    for state in self.states.values()
                    if state.circuit_open_until is not None
                ),
                default=None,
            ),
        }
        logger.warning(
            "groq_pool.unavailable %s",
            json.dumps(unavailable_diagnostic, separators=(",", ":")),
            extra=unavailable_diagnostic,
        )
        if last_error is not None:
            raise last_error
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="groq_pool",
                reason_code="provider_pool_unavailable",
                phase="provider_routing",
                endpoint=self.providers[0].client.base_url,
                model=self.providers[0].model,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                exception_class="AIProviderPoolUnavailable",
            )
        )

    async def _attempt(
        self,
        provider: GroqProvider,
        request: AIReasoningRequest,
        prompt_version: str,
        *,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> AIProviderResponse:
        last_error: AIProviderRequestError | None = None
        for attempt in range(1, self.maximum_retries + 2):
            if attempt > 1:
                self.retry_attempts += 1
            calls_before = provider.http_calls
            try:
                return await provider.reason(
                    request,
                    prompt_version=prompt_version,
                    attempt=attempt,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                )
            except AIProviderRequestError as exc:
                last_error = exc
                retryable = exc.details.reason_code in {
                    "provider_unavailable",
                    "request_timeout",
                }
                if not retryable or attempt > self.maximum_retries:
                    raise
                await asyncio.sleep(random.uniform(0.05, 0.15))
            finally:
                self._record_calls(
                    provider.provider_name,
                    provider.http_calls - calls_before,
                )
        assert last_error is not None
        raise last_error

    def _eligible(self, provider: str, now: datetime) -> bool:
        state = self.states[provider]
        self._roll_daily_metrics(state, now)
        if not state.enabled or state.status == ProviderStatus.DISABLED:
            return False
        if state.circuit_open_until is None or now >= state.circuit_open_until:
            if state.circuit_open_until is not None:
                logger.info(
                    "groq_pool.account.cooldown_completed",
                    extra={"provider": provider, "model": state.model},
                )
                state.circuit_open_until = None
                state.status = ProviderStatus.AVAILABLE
            return True
        return False

    def _record_calls(self, provider: str, count: int) -> None:
        if count <= 0:
            return
        state = self.states[provider]
        self._roll_daily_metrics(state, self.clock())
        state.calls_today += count
        self._telemetry[provider]["calls"] += count

    @staticmethod
    def _roll_daily_metrics(
        state: ProviderRuntimeState,
        now: datetime,
    ) -> None:
        current_date = now.astimezone(UTC).date().isoformat()
        if state.metrics_date == current_date:
            return
        state.metrics_date = current_date
        state.calls_today = 0
        state.successful_analyses = 0
        state.provider_failures = 0
        state.rate_limit_failures = 0
        state.quota_failures = 0
        state.input_tokens = 0
        state.output_tokens = 0
        state.total_tokens = 0

    def _success(self, provider: str, response: AIProviderResponse) -> None:
        state = self.states[provider]
        was_open = state.circuit_open_until is not None
        state.status = ProviderStatus.AVAILABLE
        state.last_success_at = self.clock()
        state.circuit_open_until = None
        state.last_failure_code = None
        state.last_provider_error_code = None
        state.last_request_result = "SCHEMA_VALID"
        state.last_attempt_schema_error = None
        state.last_success_schema_version = cast(
            str | None,
            (response.operational_metadata or {}).get(
                "analysis_schema_version"
            ),
        )
        status_code = (
            response.operational_metadata.get("status_code")
            if response.operational_metadata
            else None
        )
        state.last_http_status = status_code if isinstance(status_code, int) else 200
        usage = response.token_usage or {}
        state.input_tokens += int(usage.get("input_tokens", 0))
        state.output_tokens += int(usage.get("output_tokens", 0))
        state.total_tokens += int(usage.get("total_tokens", 0))
        self._telemetry[provider]["input_tokens"] += int(
            usage.get("input_tokens", 0)
        )
        self._telemetry[provider]["output_tokens"] += int(
            usage.get("output_tokens", 0)
        )
        self._telemetry[provider]["total_tokens"] += int(
            usage.get("total_tokens", 0)
        )
        if was_open:
            logger.info(
                "ai_provider.circuit.closed",
                extra={"provider": provider, "model": state.model},
            )

    def _failure(self, provider: str, details: AIProviderFailureDetails) -> None:
        state = self.states[provider]
        now = self.clock()
        state.last_failure_at = now
        state.last_failure_code = details.reason_code
        state.last_http_status = details.http_status
        state.last_provider_error_code = details.error_code
        state.last_request_result = details.reason_code.upper()
        state.last_attempt_schema_error = details.schema_error_code
        state.provider_failures += 1
        self._telemetry[provider]["provider_failures"] += 1
        state.recent_failures = [
            value
            for value in state.recent_failures
            if (now - value).total_seconds() <= self.circuit_rolling_window_seconds
        ]
        state.recent_failures.append(now)
        duration = self.transport_circuit_seconds
        configuration_failure = details.reason_code in {
            "authentication_failed",
            "invalid_request",
            "model_unavailable",
            "provider_unconfigured",
        }
        if details.reason_code == "authentication_failed":
            state.status = ProviderStatus.CONFIGURATION_ERROR
            duration = self.configuration_cooldown_seconds
        elif details.reason_code in {"invalid_request", "model_unavailable", "provider_unconfigured"}:
            state.status = ProviderStatus.CONFIGURATION_ERROR
            duration = self.configuration_cooldown_seconds
        elif details.reason_code == "quota_exhausted":
            state.status = ProviderStatus.QUOTA_EXHAUSTED
            state.quota_failures += 1
            self._telemetry[provider]["quota_failures"] += 1
            duration = self.quota_cooldown_seconds
        elif details.reason_code == "token_quota_exhausted":
            state.status = ProviderStatus.RATE_LIMITED
            state.rate_limit_failures += 1
            self._telemetry[provider]["rate_limit_failures"] += 1
            duration = self.rate_limit_cooldown_seconds
        elif details.reason_code == "rate_limited":
            state.status = ProviderStatus.RATE_LIMITED
            state.rate_limit_failures += 1
            self._telemetry[provider]["rate_limit_failures"] += 1
            duration = self.rate_limit_cooldown_seconds
        else:
            state.status = ProviderStatus.UNKNOWN
        should_open = (
            configuration_failure
            or details.reason_code
            in {"quota_exhausted", "token_quota_exhausted", "rate_limited"}
            or len(state.recent_failures) >= self.circuit_failure_threshold
        )
        state.circuit_open_until = (
            self._reset_at(details, now, duration) if should_open else None
        )
        if (
            state.circuit_open_until is not None
            and not configuration_failure
            and details.reason_code
            not in {"quota_exhausted", "token_quota_exhausted", "rate_limited"}
        ):
            state.status = ProviderStatus.CIRCUIT_OPEN
        logger.warning(
            "ai_provider.failure.recorded",
            extra={
                "provider": provider,
                "model": state.model,
                "status": state.status.value,
                "sanitized_error_code": details.reason_code,
                "circuit_open_until": (
                    state.circuit_open_until.isoformat()
                    if state.circuit_open_until
                    else None
                ),
                "failure_count_in_window": len(state.recent_failures),
                "circuit_failure_threshold": self.circuit_failure_threshold,
            },
        )

    def _policy_failure(
        self,
        provider: str,
        details: AIProviderFailureDetails,
    ) -> None:
        """Record request-design failure without poisoning provider eligibility."""

        state = self.states[provider]
        state.status = ProviderStatus.AVAILABLE
        state.last_http_status = details.http_status or 200
        state.last_request_result = details.reason_code.upper()
        state.last_attempt_schema_error = details.schema_error_code
        state.last_failure_code = details.reason_code
        state.last_provider_error_code = None
        state.circuit_open_until = None
        state.request_policy_failures += 1
        logger.warning(
            "ai_provider.request_policy.failed",
            extra={
                "provider": provider,
                "model": state.model,
                "status": state.status.value,
                "last_request_result": state.last_request_result,
                "provider_failure": False,
                "eligible_now": True,
            },
        )

    @staticmethod
    def _reset_at(
        details: AIProviderFailureDetails,
        now: datetime,
        default_seconds: float,
    ) -> datetime:
        raw = (
            details.rate_limit_token_reset
            if details.reason_code == "token_quota_exhausted"
            or details.rate_limit_token_remaining == "0"
            else details.rate_limit_request_reset
        ) or details.rate_limit_reset or details.retry_after
        if raw:
            try:
                value = float(raw)
                if value > now.timestamp():
                    return datetime.fromtimestamp(value, tz=UTC)
                return now + timedelta(seconds=max(1, value))
            except ValueError:
                compact = raw.strip().lower().replace(" ", "")
                matches = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", compact)
                if matches and "".join(
                    f"{number}{unit}" for number, unit in matches
                ) == compact:
                    factors = {
                        "ms": 0.001,
                        "s": 1.0,
                        "m": 60.0,
                        "h": 3600.0,
                        "d": 86400.0,
                    }
                    seconds = sum(
                        float(number) * factors[unit]
                        for number, unit in matches
                    )
                    return now + timedelta(seconds=max(1, seconds))
                try:
                    reset_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    pass
                else:
                    if reset_at.tzinfo is not None:
                        return reset_at.astimezone(UTC)
        return now + timedelta(seconds=default_seconds)

    @staticmethod
    def _failover_allowed(details: AIProviderFailureDetails) -> bool:
        if details.phase.startswith("schema_correction_"):
            return False
        if details.phase in {"request_validation", "response_decoding", "domain_parsing"}:
            return False
        return details.reason_code in {
            "authentication_failed",
            "invalid_request",
            "model_unavailable",
            "provider_unconfigured",
            "quota_exhausted",
            "token_quota_exhausted",
            "rate_limited",
            "provider_unavailable",
            "request_timeout",
        }

    def metadata(self) -> dict[str, object]:
        now = self.clock()
        available_accounts = sum(
            self._eligible(provider.provider_name, now)
            for provider in self.providers
        )
        configured_accounts = sum(
            state.enabled for state in self.states.values()
        )
        temporary_accounts = sum(
            state.status == ProviderStatus.RATE_LIMITED
            for state in self.states.values()
        )
        quota_accounts = sum(
            state.status == ProviderStatus.QUOTA_EXHAUSTED
            for state in self.states.values()
        )
        configuration_accounts = sum(
            state.status == ProviderStatus.CONFIGURATION_ERROR
            for state in self.states.values()
        )
        if available_accounts:
            aggregate_reason = "available"
        elif temporary_accounts:
            aggregate_reason = "temporarily_rate_limited"
        elif quota_accounts and quota_accounts == configured_accounts:
            aggregate_reason = "quota_exhausted"
        elif configuration_accounts == configured_accounts:
            aggregate_reason = "configuration_error"
        else:
            aggregate_reason = "unavailable"
        return {
            "provider": "groq_pool",
            "primary_provider": "Groq pool",
            "active_provider": self.active_provider,
            "latest_successful_provider": self.active_provider,
            "latest_successful_analysis_at": (
                self.latest_successful_analysis_at.isoformat()
                if self.latest_successful_analysis_at
                else None
            ),
            "model_identifier": self.providers[0].model,
            "external_ai_apis": ("groq",),
            "configured_account_count": configured_accounts,
            "available_account_count": available_accounts,
            "temporary_rate_limited_account_count": temporary_accounts,
            "quota_exhausted_account_count": quota_accounts,
            "configuration_error_account_count": configuration_accounts,
            "aggregate_reason": aggregate_reason,
            "pool_strategy": "ordered_failover",
            "providers": {
                name: state.snapshot() for name, state in self.states.items()
            },
            "call_metrics": self.metrics(),
            "circuit_policy": {
                "failure_threshold": self.circuit_failure_threshold,
                "rolling_window_seconds": self.circuit_rolling_window_seconds,
                "transport_open_duration_seconds": self.transport_circuit_seconds,
                "rate_limit_cooldown_seconds": self.rate_limit_cooldown_seconds,
                "quota_cooldown_seconds": self.quota_cooldown_seconds,
                "configuration_open_duration_seconds": self.configuration_cooldown_seconds,
                "half_open_probe": "one eligible request after open duration",
                "success_threshold_to_close": 1,
                "permanent_4xx_retried": False,
                "account_failover_failure_classes": (
                    "authentication_failed",
                    "invalid_request",
                    "model_unavailable",
                    "quota_exhausted",
                    "token_quota_exhausted",
                    "rate_limited",
                    "provider_unavailable",
                    "request_timeout",
                ),
            },
        }

    def mark_analysis_persisted(
        self,
        provider: str,
        persisted_at: datetime,
    ) -> None:
        """Select an active provider only after its validated analysis is durable."""

        self.active_provider = provider
        self.latest_successful_analysis_at = persisted_at.astimezone(UTC)
        self.states[provider].successful_analyses += 1

    def metrics(self) -> dict[str, int]:
        groq_calls = sum(provider.http_calls for provider in self.providers)
        metrics = {
            "provider_http_calls": groq_calls,
            "groq_calls": groq_calls,
            "retry_attempts": self.retry_attempts,
            "schema_corrections": (
                sum(provider.correction_attempts for provider in self.providers)
            ),
        }
        for key in (
            "analysis_requests",
            "schema_correction_requests",
            "http_429_responses",
            "initial_parse_failures",
            "initial_schema_validation_failures",
            "schema_corrections_succeeded",
            "schema_corrections_failed",
            "truncated_outputs",
            "compact_retries",
            "request_policy_failures",
            "provider_http_successes",
            "schema_valid_analyses",
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_total_tokens",
        ):
            metrics[key] = sum(
                provider.attempt_counters.get(key, 0) for provider in self.providers
            )
        for provider in self.providers:
            account_id = provider.provider_name
            telemetry = self._telemetry[account_id]
            metrics.update(
                {
                    f"{account_id}_calls": telemetry["calls"],
                    f"{account_id}_provider_failures": telemetry[
                        "provider_failures"
                    ],
                    f"{account_id}_rate_limit_failures": telemetry[
                        "rate_limit_failures"
                    ],
                    f"{account_id}_quota_failures": telemetry["quota_failures"],
                    f"{account_id}_input_tokens": telemetry["input_tokens"],
                    f"{account_id}_output_tokens": telemetry["output_tokens"],
                    f"{account_id}_total_tokens": telemetry["total_tokens"],
                }
            )
            metrics.update(
                {
                    f"{account_id}_{key}": value
                    for key, value in provider.attempt_counters.items()
                }
            )
        return metrics

    def attempts_for(self, request_id: object) -> tuple[dict[str, Any], ...]:
        attempts: list[dict[str, Any]] = []
        for provider in self.providers:
            attempts.extend(provider.attempts_for(request_id))
        return tuple(
            sorted(
                attempts,
                key=lambda item: (
                    str(item.get("recorded_at") or ""),
                    str(item.get("provider_attempt_id") or ""),
                ),
            )
        )

    def mark_model_unavailable(self, provider: str) -> None:
        selected = self.providers_by_id[provider]
        self._failure(
            provider,
            AIProviderFailureDetails(
                provider=provider,
                reason_code="model_unavailable",
                phase="startup_capability_check",
                endpoint=f"{selected.client.base_url}/models",
                model=selected.model,
                exception_class="AIProviderModelUnavailable",
            ),
        )

    def failure_snapshot(self, error: AIProviderRequestError) -> dict[str, Any]:
        return asdict(error.details)
