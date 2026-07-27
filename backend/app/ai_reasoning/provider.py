"""Provider-neutral Cerebras-primary/Groq-fallback AI reasoning boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import logging
import random
from time import perf_counter
from typing import Any, Protocol

from backend.app.ai.provider_client import (
    AIProviderClient,
    AIProviderCompletion,
    build_request_body,
    measure_request_body,
)
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.exceptions import AIProviderFailureDetails, AIProviderRequestError

from .llm_context import build_llm_analysis_context
from .models import AIReasoningRequest

logger = logging.getLogger(__name__)
AI_REASONING_RESPONSE_SCHEMA_TYPE = "ten_ai_reasoning_response"
AI_REASONING_RESPONSE_SCHEMA_VERSION = "1.0"


class ProviderStatus(StrEnum):
    HEALTHY = "HEALTHY"
    STANDBY = "STANDBY"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_FAILED = "AUTH_FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    UNCONFIGURED = "UNCONFIGURED"


@dataclass
class ProviderRuntimeState:
    status: ProviderStatus
    model: str
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    circuit_open_until: datetime | None = None
    last_failure_code: str | None = None

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
    """Strict schema accepted by both configured OpenAI-compatible providers."""

    proposal = {
        "type": ["object", "null"],
        "properties": {
            "setup_family": {"type": "string"},
            "entry_low": {"type": "number", "exclusiveMinimum": 0},
            "entry_high": {"type": "number", "exclusiveMinimum": 0},
            "stop_loss": {"type": "number", "exclusiveMinimum": 0},
            "take_profit_levels": {
                "type": "array",
                "items": {"type": "number", "exclusiveMinimum": 0},
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": [
            "setup_family",
            "entry_low",
            "entry_high",
            "stop_loss",
            "take_profit_levels",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["LONG", "SHORT", "WAIT"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string", "maxLength": 500},
            "risk_flags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "proposal": proposal,
        },
        "required": ["decision", "confidence", "rationale", "risk_flags", "proposal"],
        "additionalProperties": False,
    }


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
    ) -> AIProviderResponse:
        started = perf_counter()
        context = build_llm_analysis_context(request)
        payload = {
            "analysis_context": context.model_dump(mode="json"),
            "response_contract": self._response_contract(),
        }
        system_prompt = self.prompts.load(prompt_version)
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
            attempt=attempt,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
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
        return {
            "schema_type": AI_REASONING_RESPONSE_SCHEMA_TYPE,
            "schema_version": AI_REASONING_RESPONSE_SCHEMA_VERSION,
            "allowed_setup_families": list(self.setup_family_ids),
            "rules": [
                "return exactly one JSON object matching the supplied schema",
                "do not include chain-of-thought or private reasoning",
                "WAIT requires proposal=null",
                "LONG/SHORT requires valid ordered entry, stop, and target geometry",
                "copy setup_family exactly from allowed_setup_families",
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
    ) -> AIProviderResponse:
        try:
            return await super().reason(
                request,
                prompt_version=prompt_version,
                attempt=attempt,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        except AIProviderRequestError as exc:
            if exc.details.reason_code != "response_decoding_failed":
                raise
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
                    "sanitized_error_code": exc.details.reason_code,
                },
            )
            # One and only one bounded correction request. A second malformed
            # response propagates as a typed failure and the cycle fails closed.
            return await super().reason(
                request,
                prompt_version=prompt_version,
                attempt=attempt + 1,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
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
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.maximum_retries = maximum_retries
        self.circuit_seconds = circuit_seconds
        self.auth_circuit_seconds = auth_circuit_seconds
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
        self.active_provider = "cerebras"

    async def reason(
        self,
        request: AIReasoningRequest,
        *,
        prompt_version: str,
    ) -> AIProviderResponse:
        now = self.clock()
        fallback_reason: str | None = None
        if self._eligible("cerebras", now):
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
                if not self._fallback_allowed(exc.details):
                    raise
                self._failure("cerebras", exc.details)
                fallback_reason = f"cerebras_{exc.details.reason_code}"
        else:
            fallback_reason = (
                "cerebras_unconfigured"
                if self.states["cerebras"].status == ProviderStatus.UNCONFIGURED
                else self.states["cerebras"].last_failure_code
                or "cerebras_circuit_open"
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
        try:
            response = await self._attempt(
                self.fallback,
                request,
                prompt_version,
                fallback_used=True,
                fallback_reason=fallback_reason,
            )
        except AIProviderRequestError as exc:
            if self._fallback_allowed(exc.details):
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
        self.active_provider = "groq"
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
                retryable = exc.details.reason_code == "provider_unavailable"
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
        self.active_provider = provider
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
        duration = self.circuit_seconds
        if details.reason_code == "authentication_failed":
            state.status = ProviderStatus.AUTH_FAILED
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
        state.circuit_open_until = self._reset_at(details, now, duration)
        logger.warning(
            "ai_provider.circuit.opened",
            extra={
                "provider": provider,
                "model": state.model,
                "status": state.status.value,
                "sanitized_error_code": details.reason_code,
                "circuit_open_until": state.circuit_open_until.isoformat(),
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
            "authentication_failed",
            "quota_exhausted",
            "token_quota_exhausted",
            "rate_limited",
            "provider_unavailable",
            "model_unavailable",
        }

    def metadata(self) -> dict[str, object]:
        return {
            "provider": self.active_provider,
            "primary_provider": "cerebras",
            "active_provider": self.active_provider,
            "model_identifier": (
                self.primary.model
                if self.active_provider == "cerebras"
                else self.fallback.model
            ),
            "external_ai_apis": ("cerebras", "groq"),
            "providers": {
                name: state.snapshot() for name, state in self.states.items()
            },
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
