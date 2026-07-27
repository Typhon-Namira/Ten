"""Provider-neutral Cerebras-primary/Groq-fallback AI reasoning boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import json
import logging
import random
from time import perf_counter
from typing import Any, Protocol, cast

from pydantic import ValidationError

from backend.app.ai.provider_client import (
    AIProviderClient,
    AIProviderCompletion,
    build_request_body,
    measure_request_body,
)
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.exceptions import AIProviderFailureDetails, AIProviderRequestError

from .llm_context import build_llm_analysis_context
from .analysis import AIAnalysisOutput
from .models import AIReasoningRequest

logger = logging.getLogger(__name__)
AI_REASONING_RESPONSE_SCHEMA_TYPE = "ten_ai_reasoning_response"
AI_REASONING_RESPONSE_SCHEMA_VERSION = "1.0"
_MAX_PROVIDER_JSON_LOG_CHARACTERS = 8_000


def _bounded_provider_json(value: str) -> tuple[str, bool]:
    """Bound compact provider output diagnostics without logging prompts or credentials."""

    return (
        value[:_MAX_PROVIDER_JSON_LOG_CHARACTERS],
        len(value) > _MAX_PROVIDER_JSON_LOG_CHARACTERS,
    )


class ProviderStatus(StrEnum):
    HEALTHY = "HEALTHY"
    STANDBY = "STANDBY"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_FAILED = "AUTH_FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    UNCONFIGURED = "UNCONFIGURED"


@dataclass
class ProviderRuntimeState:
    status: ProviderStatus
    model: str
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    circuit_open_until: datetime | None = None
    last_failure_code: str | None = None
    last_http_status: int | None = None
    last_provider_error_code: str | None = None
    recent_failures: list[datetime] = field(default_factory=list)

    def snapshot(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "model": self.model,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "circuit_open_until": (
                self.circuit_open_until.isoformat() if self.circuit_open_until else None
            ),
            "last_failure_code": self.last_failure_code,
            "last_http_status": self.last_http_status,
            "last_provider_error_code": self.last_provider_error_code,
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


def reasoning_response_schema() -> dict[str, Any]:
    """Strict analysis-only schema accepted by both configured providers."""

    unsupported_keywords = {
        "title",
        "default",
        "description",
        "format",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        # Numeric bounds remain enforced by the unchanged Pydantic
        # domain schema after decoding. Omitting them from the wire
        # contract keeps Cerebras below its 5,000-character schema
        # ceiling without weakening application validation.
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

    return cast(dict[str, Any], compact(AIAnalysisOutput.model_json_schema()))


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
    ) -> AIProviderResponse:
        started = perf_counter()
        context = build_llm_analysis_context(request)
        payload = {
            "analysis_context": context.model_dump(mode="json"),
            "response_contract": self._response_contract(),
        }
        system_prompt = self.prompts.load(prompt_version)
        if correction_instruction:
            system_prompt = (
                f"{system_prompt}\n\n"
                "CORRECTION REQUIRED FOR THE PREVIOUS RESPONSE:\n"
                f"{correction_instruction}\n"
                "Return one complete JSON object containing every required field. "
                "Do not omit fields and do not add properties outside the response contract."
            )
        schema = reasoning_response_schema()
        wire_schema = schema if self.supports_strict_json_schema else None
        request_body = build_request_body(
            system_prompt=system_prompt,
            payload=payload,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            },
        )
        rejection_reason: str | None = None
        if metrics.maximum_output_tokens > self.absolute_max_output_tokens:
            rejection_reason = "maximum_output_tokens_exceeded"
        elif metrics.estimated_input_tokens > self.hard_input_tokens:
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
                        f"estimated_input_tokens={metrics.estimated_input_tokens} "
                        f"maximum_output_tokens={metrics.maximum_output_tokens}"
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
        attempt_type = (
            "correction"
            if correction_instruction
            else "fallback"
            if fallback_used
            else "retry"
            if attempt > 1
            else "primary"
        )
        completion = await self.client.complete_json(
            system_prompt=system_prompt,
            payload=payload,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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
            attempt_type=attempt_type,
        )
        return self._response(completion, started, fallback_used, fallback_reason)

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
    ) -> AIProviderResponse:
        decoded_json = json.dumps(
            completion.content,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw_json, raw_truncated = _bounded_provider_json(
            completion.raw_json_text or decoded_json
        )
        decoded_json, decoded_truncated = _bounded_provider_json(decoded_json)
        logger.info(
            "ai_provider.response.raw",
            extra={
                "provider": self.provider_name,
                "model": self.model,
                "provider_request_id": completion.provider_request_id,
                "status_code": completion.status_code,
                "raw_provider_json": raw_json,
                "raw_provider_json_truncated": raw_truncated,
                "decoded_provider_json": decoded_json,
                "decoded_provider_json_truncated": decoded_truncated,
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
            },
        )

    def _response_contract(self) -> dict[str, Any]:
        schema = reasoning_response_schema()
        return {
            # Groq JSON Object Mode does not receive response_format.json_schema,
            # so the complete schema must be present in the prompt payload.
            "json_schema": schema,
            "rules": [
                "return exactly one JSON object matching the supplied schema",
                "return no markdown and no prose outside the JSON object",
                "do not include chain-of-thought or private reasoning",
                "do not recommend BUY, SELL, WAIT, LONG, SHORT, HOLD, or any trading action",
                "do not emit a proposal, setup family, entry, stop loss, target, readiness, or publication decision",
                "reference only evidence present in analysis_context and never invent evidence",
            ],
        }


class CerebrasProvider(_OpenAICompatibleReasoningProvider):
    provider_name = "cerebras"


class GroqProvider(_OpenAICompatibleReasoningProvider):
    provider_name = "groq"
    # llama-3.1-8b-instant supports JSON Object Mode, but not Groq's strict
    # json_schema constrained-decoding mode. TEN's unchanged application validator
    # remains the authoritative schema boundary.
    supports_strict_json_schema = False

    async def reason(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
        attempt: int = 1,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        correction_instruction: str | None = None,
    ) -> AIProviderResponse:
        try:
            response = await super().reason(
                request,
                prompt_version=prompt_version,
                attempt=attempt,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                correction_instruction=correction_instruction,
            )
        except AIProviderRequestError as exc:
            if exc.details.reason_code != "response_decoding_failed":
                raise
            correction_instruction = (
                "The previous response was not valid JSON. Return valid JSON matching "
                "the complete response contract."
            )
            correction_reason = exc.details.reason_code
        else:
            try:
                AIAnalysisOutput.model_validate(response.raw_output)
                return response
            except ValidationError as exc:
                errors = exc.errors()
                if errors:
                    first = errors[0]
                    field_path = ".".join(
                        (
                            "provider_response",
                            *(str(item) for item in first["loc"]),
                        )
                    )
                    expected = str(first["msg"] or first["type"])
                    error_type = str(first["type"])
                else:
                    field_path = "provider_response"
                    expected = "schema validation"
                    error_type = "schema_validation"
                if error_type == "extra_forbidden":
                    correction_instruction = (
                        f"The previous JSON contained an unexpected property at {field_path}: "
                        f"{expected}. Remove that property. Return only properties declared in "
                        "response_contract.json_schema. Do not emit contract metadata such as "
                        "schema_type, schema_version, or json_schema."
                    )
                else:
                    correction_instruction = (
                        f"The previous JSON failed schema validation at {field_path}: {expected}. "
                        "Return that exact missing field and every required property in json_schema. "
                        "Return analysis only and do not add signal, proposal, execution, or publication fields."
                    )
                correction_reason = "structured_output_invalid"

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
            },
        )
        # One and only one bounded correction request. A second malformed or
        # schema-invalid response is returned to the application validator, which
        # persists the typed failure and keeps publication fail-closed.
        self.correction_attempts += 1
        return await super().reason(
            request,
            prompt_version=prompt_version,
            attempt=attempt + 1,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            correction_instruction=correction_instruction,
        )


class AIProviderRouter:
    """Sequential primary/fallback router with bounded retries and provider circuits."""

    def __init__(
        self,
        primary: CerebrasProvider,
        fallback: GroqProvider,
        *,
        maximum_retries: int = 1,
        circuit_seconds: float = 300,
        auth_circuit_seconds: float = 3600,
        circuit_failure_threshold: int = 2,
        circuit_rolling_window_seconds: float = 60,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.maximum_retries = maximum_retries
        self.circuit_seconds = circuit_seconds
        self.auth_circuit_seconds = auth_circuit_seconds
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_rolling_window_seconds = circuit_rolling_window_seconds
        self.clock = clock or (lambda: datetime.now(UTC))
        self.states = {
            "cerebras": ProviderRuntimeState(
                ProviderStatus.STANDBY if primary.configured else ProviderStatus.UNCONFIGURED,
                primary.model,
            ),
            "groq": ProviderRuntimeState(
                ProviderStatus.STANDBY if fallback.configured else ProviderStatus.UNCONFIGURED,
                fallback.model,
            ),
        }
        self.active_provider: str | None = None
        self.latest_successful_analysis_at: datetime | None = None
        self.retry_attempts = 0
        self.fallback_attempts = 0

    async def reason(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
    ) -> AIProviderResponse:
        now = self.clock()
        fallback_reason: str | None = None
        primary_eligible = self._eligible("cerebras", now)
        logger.info(
            "ai_provider.routing.entered",
            extra={
                "provider": "cerebras",
                "model": self.primary.model,
                "instrument": request.instrument,
                "cycle_id": str(request.cycle_id),
                "ums_boundary": request.analysis_timestamp.isoformat(),
                "primary_provider": "cerebras",
                "primary_eligible": primary_eligible,
                "primary_status": self.states["cerebras"].status.value,
                "primary_circuit_open_until": (
                    self.states["cerebras"].circuit_open_until.isoformat()
                    if self.states["cerebras"].circuit_open_until
                    else None
                ),
                "fallback_status": self.states["groq"].status.value,
            },
        )
        if primary_eligible:
            logger.info(
                "ai_provider.primary.started",
                extra={
                    "provider": "cerebras",
                    "model": self.primary.model,
                    "instrument": request.instrument,
                    "cycle_id": str(request.cycle_id),
                    "ums_boundary": request.analysis_timestamp.isoformat(),
                },
            )
            try:
                response = await self._attempt(
                    self.primary,
                    request,
                    prompt_version,
                    fallback_used=False,
                    fallback_reason=None,
                )
                self._success("cerebras")
                return response
            except AIProviderRequestError as exc:
                self._failure("cerebras", exc.details)
                logger.warning(
                    "ai_provider.primary.failed",
                    extra={
                        "provider": "cerebras",
                        "model": self.primary.model,
                        "instrument": request.instrument,
                        "cycle_id": str(request.cycle_id),
                        "ums_boundary": request.analysis_timestamp.isoformat(),
                        "status_code": exc.details.http_status,
                        "sanitized_error_code": exc.details.reason_code,
                        "fallback_allowed": self._fallback_allowed(exc.details),
                    },
                )
                if not self._fallback_allowed(exc.details):
                    raise
                fallback_reason = f"cerebras_{exc.details.reason_code}"
        else:
            primary_state = self.states["cerebras"]
            fallback_reason = (
                "cerebras_unconfigured"
                if primary_state.status == ProviderStatus.UNCONFIGURED
                else (
                    f"cerebras_{primary_state.last_failure_code}"
                    if primary_state.last_failure_code
                    else None
                )
                or "cerebras_circuit_open"
            )
            logger.warning(
                "ai_provider.primary.skipped",
                extra={
                    "provider": "cerebras",
                    "model": self.primary.model,
                    "instrument": request.instrument,
                    "cycle_id": str(request.cycle_id),
                    "ums_boundary": request.analysis_timestamp.isoformat(),
                    "skip_reason": fallback_reason,
                    "primary_status": self.states["cerebras"].status.value,
                    "circuit_open_until": (
                        self.states["cerebras"].circuit_open_until.isoformat()
                        if self.states["cerebras"].circuit_open_until
                        else None
                    ),
                },
            )
            if primary_state.status == ProviderStatus.UNCONFIGURED or (
                primary_state.last_failure_code
                not in {"rate_limited", "provider_unavailable", "request_timeout"}
            ):
                raise AIProviderRequestError(
                    AIProviderFailureDetails(
                        provider="cerebras",
                        reason_code=(
                            primary_state.last_failure_code
                            or "provider_unconfigured"
                        ),
                        phase="provider_routing",
                        endpoint=f"{self.primary.client.base_url}/chat/completions",
                        model=self.primary.model,
                        request_id=str(request.request_id),
                        cycle_id=str(request.cycle_id),
                        exception_class="AIProviderRoutingError",
                    )
                )

        logger.info(
            "ai_provider.fallback.started",
            extra={
                "provider": "groq",
                "model": self.fallback.model,
                "instrument": request.instrument,
                "cycle_id": str(request.cycle_id),
                "ums_boundary": request.analysis_timestamp.isoformat(),
                "fallback_used": True,
                "fallback_reason": fallback_reason,
            },
        )
        if not self._eligible("groq", now):
            details = AIProviderFailureDetails(
                provider="groq",
                reason_code=(
                    "provider_unconfigured"
                    if self.states["groq"].status == ProviderStatus.UNCONFIGURED
                    else "provider_circuit_open"
                ),
                phase="provider_routing",
                endpoint=f"{self.fallback.client.base_url}/chat/completions",
                model=self.fallback.model,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                fallback_used=True,
                fallback_reason=fallback_reason,
                exception_class="AIProviderRoutingError",
            )
            raise AIProviderRequestError(details)
        self.fallback_attempts += 1
        try:
            response = await self._attempt(
                self.fallback,
                request,
                prompt_version,
                fallback_used=True,
                fallback_reason=fallback_reason,
            )
        except AIProviderRequestError as exc:
            self._failure("groq", exc.details)
            details = replace(
                exc.details,
                fallback_used=True,
                fallback_reason=fallback_reason,
            )
            logger.error(
                "ai_provider.fallback.completed",
                extra={
                    "provider": "groq",
                    "model": self.fallback.model,
                    "instrument": request.instrument,
                    "cycle_id": str(request.cycle_id),
                    "ums_boundary": request.analysis_timestamp.isoformat(),
                    "fallback_used": True,
                    "fallback_reason": fallback_reason,
                    "provider_status": "failed",
                    "reasoning_status": "degraded",
                    "publication_eligible": False,
                    "sanitized_error_code": details.reason_code,
                },
            )
            raise AIProviderRequestError(details) from exc
        self._success("groq")
        logger.info(
            "ai_provider.fallback.completed",
            extra={
                "provider": "groq",
                "model": self.fallback.model,
                "instrument": request.instrument,
                "cycle_id": str(request.cycle_id),
                "ums_boundary": request.analysis_timestamp.isoformat(),
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "provider_status": "success",
                "reasoning_status": "valid",
            },
        )
        return response

    async def _attempt(
        self,
        provider: _OpenAICompatibleReasoningProvider,
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
        assert last_error is not None
        raise last_error

    def _eligible(self, provider: str, now: datetime) -> bool:
        state = self.states[provider]
        if state.status == ProviderStatus.UNCONFIGURED:
            return False
        if state.circuit_open_until is None or now >= state.circuit_open_until:
            if state.circuit_open_until is not None:
                logger.info(
                    "ai_provider.circuit.closed",
                    extra={"provider": provider, "model": state.model},
                )
                state.circuit_open_until = None
                state.status = ProviderStatus.STANDBY
            return True
        return False

    def _success(self, provider: str) -> None:
        state = self.states[provider]
        was_open = state.circuit_open_until is not None
        state.status = ProviderStatus.HEALTHY
        state.last_success_at = self.clock()
        state.circuit_open_until = None
        state.last_failure_code = None
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
        state.recent_failures = [
            value
            for value in state.recent_failures
            if (now - value).total_seconds() <= self.circuit_rolling_window_seconds
        ]
        state.recent_failures.append(now)
        duration = self.circuit_seconds
        configuration_failure = details.reason_code in {
            "authentication_failed",
            "invalid_request",
            "model_unavailable",
            "provider_unconfigured",
        }
        if details.reason_code == "authentication_failed":
            state.status = ProviderStatus.CONFIGURATION_ERROR
            duration = self.auth_circuit_seconds
        elif details.reason_code in {"invalid_request", "model_unavailable", "provider_unconfigured"}:
            state.status = ProviderStatus.CONFIGURATION_ERROR
            duration = self.auth_circuit_seconds
        elif details.reason_code == "quota_exhausted":
            state.status = ProviderStatus.QUOTA_EXHAUSTED
            duration = max(self.circuit_seconds, 3600)
        elif details.reason_code == "token_quota_exhausted":
            state.status = ProviderStatus.RATE_LIMITED
        elif details.reason_code == "rate_limited":
            state.status = ProviderStatus.RATE_LIMITED
        else:
            state.status = ProviderStatus.UNAVAILABLE
        should_open = (
            configuration_failure
            or details.reason_code
            in {"quota_exhausted", "token_quota_exhausted", "rate_limited"}
            or len(state.recent_failures) >= self.circuit_failure_threshold
        )
        state.circuit_open_until = (
            self._reset_at(details, now, duration) if should_open else None
        )
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

    @staticmethod
    def _reset_at(
        details: AIProviderFailureDetails,
        now: datetime,
        default_seconds: float,
    ) -> datetime:
        raw = (
            details.rate_limit_token_reset
            if details.reason_code == "token_quota_exhausted"
            else details.rate_limit_request_reset
        ) or details.rate_limit_reset or details.retry_after
        if raw:
            try:
                value = float(raw)
                if value > now.timestamp():
                    return datetime.fromtimestamp(value, tz=UTC)
                return now + timedelta(seconds=max(1, value))
            except ValueError:
                pass
        return now + timedelta(seconds=default_seconds)

    @staticmethod
    def _fallback_allowed(details: AIProviderFailureDetails) -> bool:
        if details.phase in {"request_validation", "response_decoding", "domain_parsing"}:
            return False
        return details.reason_code in {
            "rate_limited",
            "provider_unavailable",
            "request_timeout",
        }

    def metadata(self) -> dict[str, object]:
        return {
            "provider": self.active_provider,
            "primary_provider": "cerebras",
            "active_provider": self.active_provider,
            "latest_successful_provider": self.active_provider,
            "latest_successful_analysis_at": (
                self.latest_successful_analysis_at.isoformat()
                if self.latest_successful_analysis_at
                else None
            ),
            "model_identifier": (
                self.primary.model
                if self.active_provider == "cerebras"
                else self.fallback.model
                if self.active_provider == "groq"
                else self.primary.model
            ),
            "external_ai_apis": ("cerebras", "groq"),
            "providers": {
                name: state.snapshot() for name, state in self.states.items()
            },
            "call_metrics": self.metrics(),
            "circuit_policy": {
                "failure_threshold": self.circuit_failure_threshold,
                "rolling_window_seconds": self.circuit_rolling_window_seconds,
                "open_duration_seconds": self.circuit_seconds,
                "configuration_open_duration_seconds": self.auth_circuit_seconds,
                "half_open_probe": "one eligible request after open duration",
                "success_threshold_to_close": 1,
                "permanent_4xx_retried": False,
                "fallback_failure_classes": (
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

    def metrics(self) -> dict[str, int]:
        cerebras_calls = self.primary.http_calls
        groq_calls = self.fallback.http_calls
        return {
            "provider_http_calls": cerebras_calls + groq_calls,
            "cerebras_calls": cerebras_calls,
            "groq_fallback_calls": groq_calls,
            "retry_attempts": self.retry_attempts,
            "fallback_attempts": self.fallback_attempts,
            "schema_corrections": (
                self.primary.correction_attempts
                + self.fallback.correction_attempts
            ),
        }

    def mark_model_unavailable(self, provider: str) -> None:
        selected = self.primary if provider == "cerebras" else self.fallback
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
