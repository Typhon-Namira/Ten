"""Provider-neutral OpenAI-compatible transport for Cerebras and Groq."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import json
import logging
from time import perf_counter
from typing import Any, Protocol

import httpx

from backend.app.core.exceptions import (
    AIProviderFailureDetails,
    AIProviderRequestError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


def _safe_text(value: object, *, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] or None


def _normalized_code(value: str | None) -> str | None:
    if not value:
        return None
    normalized = "".join(
        character if character.isalnum() else "_" for character in value.lower()
    ).strip("_")
    return normalized[:96] or None


def _reason_code(
    status: int | None,
    provider_code: str | None,
    error_message: str | None,
    metadata_error_type: str | None,
) -> str:
    error_text = " ".join(
        value.lower()
        for value in (provider_code, error_message, metadata_error_type)
        if value
    )
    if "context" in error_text and ("limit" in error_text or "length" in error_text):
        return "context_limit_exceeded"
    if "token" in error_text and any(
        marker in error_text
        for marker in ("quota", "rate limit", "limit reached", "limit exceeded", "exhausted")
    ):
        return "token_quota_exhausted"
    if any(token in error_text for token in ("daily quota", "project quota", "quota exhausted")):
        return "quota_exhausted"
    if "quota" in error_text or "rate limit" in error_text:
        return "rate_limited"
    if status is not None and 500 <= status <= 599:
        return "provider_unavailable"
    known = {
        400: "invalid_request",
        401: "authentication_failed",
        402: "quota_exhausted",
        403: "authentication_failed",
        404: "model_unavailable",
        413: "request_too_large",
        429: "rate_limited",
    }
    if status in known:
        return known[status]
    return _normalized_code(provider_code) or (
        "invalid_request" if status is not None else "provider_unavailable"
    )


def _error_fields(
    response: httpx.Response,
) -> tuple[str | None, str | None, str | None, str | None]:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None, None, None, None
    if not isinstance(body, dict):
        return None, None, None, None
    error = body.get("error")
    if isinstance(error, str):
        return None, _safe_text(error), None, None
    if not isinstance(error, dict):
        return None, None, None, None
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return (
        _safe_text(error.get("code"), limit=96),
        _safe_text(error.get("message")),
        _safe_text(metadata.get("error_type"), limit=128),
        _safe_text(metadata.get("provider_code"), limit=128),
    )


@dataclass(frozen=True)
class AIProviderRequestMetrics:
    serialized_request_bytes: int
    estimated_input_tokens: int
    maximum_output_tokens: int
    estimated_maximum_cost_usd: float
    system_prompt_characters: int
    user_prompt_characters: int
    message_count: int
    response_schema_bytes: int


@dataclass(frozen=True)
class AIProviderCompletion:
    content: dict[str, Any]
    provider: str
    model: str
    status_code: int
    latency_ms: float
    provider_request_id: str | None
    token_usage: dict[str, int] | None
    rate_limit_limit: str | None
    rate_limit_remaining: str | None
    rate_limit_reset: str | None
    rate_limit_request_limit: str | None = None
    rate_limit_request_remaining: str | None = None
    rate_limit_request_reset: str | None = None
    rate_limit_token_limit: str | None = None
    rate_limit_token_remaining: str | None = None
    rate_limit_token_reset: str | None = None
    retry_after: str | None = None
    raw_json_text: str | None = None


class AIProviderClient(Protocol):
    provider: str
    base_url: str

    @property
    def configured(self) -> bool: ...

    async def available_models(self) -> tuple[str, ...]: ...

    async def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        model: str,
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        request_id: str | None = None,
        cycle_id: str | None = None,
        instrument: str | None = None,
        ums_boundary: str | None = None,
        attempt: int = 1,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> AIProviderCompletion: ...


def build_request_body(
    *,
    system_prompt: str,
    payload: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_format: dict[str, Any]
    if response_schema is None:
        response_format = {"type": "json_object"}
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "ten_ai_reasoning_response",
                "strict": True,
                "schema": response_schema,
            },
        }
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def measure_request_body(
    body: dict[str, Any],
    *,
    input_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
) -> AIProviderRequestMetrics:
    encoded = httpx.Request("POST", "https://provider.invalid", json=body).content
    messages = body.get("messages", ())
    system_chars = sum(
        len(str(item.get("content", "")))
        for item in messages
        if isinstance(item, dict) and item.get("role") == "system"
    )
    user_chars = sum(
        len(str(item.get("content", "")))
        for item in messages
        if isinstance(item, dict) and item.get("role") == "user"
    )
    estimated_input = ceil(len(encoded) / 3)
    maximum_output = int(body.get("max_tokens", 0))
    maximum_cost = (
        estimated_input * input_cost_per_million_usd
        + maximum_output * output_cost_per_million_usd
    ) / 1_000_000
    return AIProviderRequestMetrics(
        serialized_request_bytes=len(encoded),
        estimated_input_tokens=estimated_input,
        maximum_output_tokens=maximum_output,
        estimated_maximum_cost_usd=maximum_cost,
        system_prompt_characters=system_chars,
        user_prompt_characters=user_chars,
        message_count=len(messages),
        response_schema_bytes=len(
            json.dumps(body.get("response_format", {}), separators=(",", ":")).encode()
        ),
    )


class HttpAIProviderClient:
    """One explicitly configured provider client with no hidden retries."""

    def __init__(
        self,
        provider: str,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float = 30.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    async def available_models(self) -> tuple[str, ...]:
        if not self.api_key:
            return ()
        endpoint = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.get(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                data = response.json().get("data", ())
        except (httpx.HTTPError, ValueError, AttributeError):
            return ()
        return tuple(
            str(item["id"])
            for item in data
            if isinstance(item, dict) and item.get("id")
        )

    async def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        model: str,
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        request_id: str | None = None,
        cycle_id: str | None = None,
        instrument: str | None = None,
        ums_boundary: str | None = None,
        attempt: int = 1,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> AIProviderCompletion:
        if not self.api_key:
            raise ConfigurationError(
                f"TEN_{self.provider.upper()}_API_KEY is required for {self.provider}"
            )
        endpoint = f"{self.base_url}/chat/completions"
        body = build_request_body(
            system_prompt=system_prompt,
            payload=payload,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )
        metrics = measure_request_body(
            body,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )
        context = {
            "provider": self.provider,
            "model": model,
            "instrument": instrument,
            "cycle_id": cycle_id,
            "ums_boundary": ums_boundary,
            "attempt": attempt,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "request_id": request_id,
            "endpoint": endpoint,
            "status_code": None,
            "error_type": None,
            "sanitized_error_code": None,
            "latency_ms": None,
            "provider_request_id": None,
            "rate_limit_remaining": None,
            "rate_limit_reset": None,
            "exception_type": None,
            "network_error_category": None,
        }
        logger.info(
            "ai_provider.request.started",
            extra={
                **context,
                "serialized_request_bytes": metrics.serialized_request_bytes,
                "estimated_input_tokens": metrics.estimated_input_tokens,
            },
        )
        started = perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self._http_failure(
                exc.response,
                model=model,
                request_id=request_id,
                cycle_id=cycle_id,
                instrument=instrument,
                ums_boundary=ums_boundary,
                attempt=attempt,
                started=started,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            ) from exc
        except httpx.RequestError as exc:
            details = AIProviderFailureDetails(
                provider=self.provider,
                reason_code="provider_unavailable",
                phase="http_request",
                endpoint=endpoint,
                model=model,
                request_id=request_id,
                cycle_id=cycle_id,
                error_message=_safe_text(str(exc)),
                elapsed_ms=(perf_counter() - started) * 1000,
                exception_class=type(exc).__name__,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            self._log_failure(details, instrument=instrument, ums_boundary=ums_boundary, attempt=attempt)
            raise AIProviderRequestError(details) from exc

        headers = response.headers
        elapsed = (perf_counter() - started) * 1000
        provider_request_id = _safe_text(
            headers.get("x-request-id") or headers.get("request-id"),
            limit=128,
        )
        request_limit = _safe_text(headers.get("x-ratelimit-limit-requests"))
        request_remaining = _safe_text(headers.get("x-ratelimit-remaining-requests"))
        request_reset = _safe_text(headers.get("x-ratelimit-reset-requests"))
        token_limit = _safe_text(headers.get("x-ratelimit-limit-tokens"))
        token_remaining = _safe_text(headers.get("x-ratelimit-remaining-tokens"))
        token_reset = _safe_text(headers.get("x-ratelimit-reset-tokens"))
        rate_limit_remaining = request_remaining or _safe_text(
            headers.get("x-ratelimit-remaining")
        )
        rate_limit_reset = request_reset or _safe_text(headers.get("x-ratelimit-reset"))
        response_context = {
            **context,
            "status_code": response.status_code,
            "latency_ms": elapsed,
            "provider_request_id": provider_request_id,
            "rate_limit_remaining": rate_limit_remaining,
            "rate_limit_reset": rate_limit_reset,
        }
        logger.info("ai_provider.request.http_completed", extra=response_context)
        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
        except (KeyError, IndexError, json.JSONDecodeError, TypeError, ValueError) as exc:
            details = AIProviderFailureDetails(
                provider=self.provider,
                reason_code="response_decoding_failed",
                phase="response_decoding",
                endpoint=endpoint,
                model=model,
                request_id=request_id,
                cycle_id=cycle_id,
                http_status=response.status_code,
                content_type=_safe_text(headers.get("content-type"), limit=128),
                body_length=len(response.content),
                provider_request_id=provider_request_id,
                elapsed_ms=elapsed,
                exception_class=type(exc).__name__,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            self._log_failure(details, instrument=instrument, ums_boundary=ums_boundary, attempt=attempt)
            raise AIProviderRequestError(details) from exc
        if not isinstance(parsed, dict):
            details = AIProviderFailureDetails(
                provider=self.provider,
                reason_code="domain_parsing_failed",
                phase="domain_parsing",
                endpoint=endpoint,
                model=model,
                request_id=request_id,
                cycle_id=cycle_id,
                http_status=response.status_code,
                body_length=len(response.content),
                provider_request_id=provider_request_id,
                elapsed_ms=elapsed,
                exception_class="TypeError",
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            self._log_failure(details, instrument=instrument, ums_boundary=ums_boundary, attempt=attempt)
            raise AIProviderRequestError(details)
        usage = envelope.get("usage")
        token_usage = (
            {
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
            if isinstance(usage, dict)
            else None
        )
        logger.info("ai_provider.response.received", extra=response_context)
        return AIProviderCompletion(
            content=parsed,
            provider=self.provider,
            model=model,
            status_code=response.status_code,
            latency_ms=elapsed,
            provider_request_id=provider_request_id,
            token_usage=token_usage,
            rate_limit_limit=_safe_text(
                request_limit or headers.get("x-ratelimit-limit")
            ),
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset=rate_limit_reset,
            rate_limit_request_limit=request_limit,
            rate_limit_request_remaining=request_remaining,
            rate_limit_request_reset=request_reset,
            rate_limit_token_limit=token_limit,
            rate_limit_token_remaining=token_remaining,
            rate_limit_token_reset=token_reset,
            retry_after=_safe_text(headers.get("retry-after")),
            raw_json_text=(
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            ),
        )

    def _http_failure(
        self,
        response: httpx.Response,
        *,
        model: str,
        request_id: str | None,
        cycle_id: str | None,
        instrument: str | None,
        ums_boundary: str | None,
        attempt: int,
        started: float,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> AIProviderRequestError:
        error_code, error_message, metadata_error_type, metadata_provider_code = (
            _error_fields(response)
        )
        headers = response.headers
        request_limit = _safe_text(headers.get("x-ratelimit-limit-requests"))
        request_remaining = _safe_text(headers.get("x-ratelimit-remaining-requests"))
        request_reset = _safe_text(headers.get("x-ratelimit-reset-requests"))
        token_limit = _safe_text(headers.get("x-ratelimit-limit-tokens"))
        token_remaining = _safe_text(headers.get("x-ratelimit-remaining-tokens"))
        token_reset = _safe_text(headers.get("x-ratelimit-reset-tokens"))
        details = AIProviderFailureDetails(
            provider=self.provider,
            reason_code=_reason_code(
                response.status_code,
                error_code,
                error_message,
                metadata_error_type,
            ),
            phase="http_request",
            endpoint=f"{self.base_url}/chat/completions",
            model=model,
            request_id=request_id,
            cycle_id=cycle_id,
            http_status=response.status_code,
            error_code=error_code,
            error_message=error_message,
            metadata_error_type=metadata_error_type,
            metadata_provider_code=metadata_provider_code,
            content_type=_safe_text(headers.get("content-type"), limit=128),
            body_length=len(response.content),
            retry_after=_safe_text(headers.get("retry-after")),
            rate_limit_limit=_safe_text(
                request_limit or headers.get("x-ratelimit-limit")
            ),
            rate_limit_remaining=_safe_text(
                request_remaining or headers.get("x-ratelimit-remaining")
            ),
            rate_limit_reset=_safe_text(
                request_reset or headers.get("x-ratelimit-reset")
            ),
            rate_limit_request_limit=request_limit,
            rate_limit_request_remaining=request_remaining,
            rate_limit_request_reset=request_reset,
            rate_limit_token_limit=token_limit,
            rate_limit_token_remaining=token_remaining,
            rate_limit_token_reset=token_reset,
            provider_request_id=_safe_text(
                headers.get("x-request-id") or headers.get("request-id"),
                limit=128,
            ),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            elapsed_ms=(perf_counter() - started) * 1000,
            exception_class="HTTPStatusError",
        )
        self._log_failure(
            details,
            instrument=instrument,
            ums_boundary=ums_boundary,
            attempt=attempt,
        )
        return AIProviderRequestError(details)

    @staticmethod
    def _log_failure(
        details: AIProviderFailureDetails,
        *,
        instrument: str | None = None,
        ums_boundary: str | None = None,
        attempt: int = 1,
    ) -> None:
        logger.error(
            "ai_provider.request.failed",
            extra={
                "provider": details.provider,
                "request_id": details.request_id,
                "model": details.model,
                "instrument": instrument,
                "cycle_id": details.cycle_id,
                "ums_boundary": ums_boundary,
                "attempt": attempt,
                "fallback_used": details.fallback_used,
                "fallback_reason": details.fallback_reason,
                "status_code": details.http_status,
                "error_type": details.metadata_error_type,
                "sanitized_error_code": details.reason_code,
                "latency_ms": details.elapsed_ms,
                "provider_request_id": details.provider_request_id,
                "rate_limit_remaining": details.rate_limit_remaining,
                "rate_limit_reset": details.rate_limit_reset,
                "rate_limit_request_limit": details.rate_limit_request_limit,
                "rate_limit_request_remaining": details.rate_limit_request_remaining,
                "rate_limit_request_reset": details.rate_limit_request_reset,
                "rate_limit_token_limit": details.rate_limit_token_limit,
                "rate_limit_token_remaining": details.rate_limit_token_remaining,
                "rate_limit_token_reset": details.rate_limit_token_reset,
                "exception_type": details.exception_class,
                "safe_exception_message": details.error_message,
                "network_error_category": (
                    details.exception_class
                    if details.phase == "http_request"
                    and details.http_status is None
                    else None
                ),
            },
        )
