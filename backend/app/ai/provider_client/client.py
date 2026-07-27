"""Provider-neutral OpenAI-compatible transport used by the Groq account pool."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import json
import logging
import re
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from backend.app.core.exceptions import (
    AIProviderFailureDetails,
    AIProviderRequestError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)

RATE_LIMITED_TEMPORARY = "RATE_LIMITED_TEMPORARY"
DAILY_TOKEN_QUOTA_EXHAUSTED = "DAILY_TOKEN_QUOTA_EXHAUSTED"
REQUEST_TOKEN_LIMIT_EXCEEDED = "REQUEST_TOKEN_LIMIT_EXCEEDED"
CONCURRENT_LIMIT = "CONCURRENT_LIMIT"
ACCOUNT_CONFIGURATION_ERROR = "ACCOUNT_CONFIGURATION_ERROR"
UNKNOWN_PROVIDER_LIMIT = "UNKNOWN_PROVIDER_LIMIT"


class ProviderJSONDecodeError(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def extract_single_json_object(content: object) -> tuple[dict[str, Any], str | None]:
    """Extract one unambiguous object without repairing its business values."""

    if isinstance(content, dict):
        return content, None
    if not isinstance(content, str):
        raise ProviderJSONDecodeError("json_not_found")
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    extraction_note: str | None = None
    if fenced:
        stripped = fenced.group(1).strip()
        extraction_note = "markdown_fence_present"
    first = stripped.find("{")
    if first < 0:
        raise ProviderJSONDecodeError("json_not_found")
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(stripped, first)
    except json.JSONDecodeError as exc:
        reason = (
            "truncated_response"
            if exc.pos >= max(first, len(stripped) - 2)
            else "json_parse_error"
        )
        raise ProviderJSONDecodeError(reason) from exc
    if not isinstance(parsed, dict):
        raise ProviderJSONDecodeError("json_not_found")
    suffix = stripped[end:].strip()
    if suffix:
        next_object = suffix.find("{")
        if next_object >= 0:
            try:
                second, _ = decoder.raw_decode(suffix, next_object)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(second, dict):
                    raise ProviderJSONDecodeError("multiple_json_objects")
        extraction_note = extraction_note or "surrounding_prose_removed"
    if first > 0:
        extraction_note = extraction_note or "surrounding_prose_removed"
    return parsed, extraction_note


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
    if any(
        marker in error_text
        for marker in (
            "daily quota",
            "daily token",
            "daily request",
            "per day",
            "tokens per day",
            "requests per day",
            "project quota",
            "quota exhausted",
        )
    ):
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
        408: "request_timeout",
        413: "request_too_large",
        429: "rate_limited",
    }
    if status in known:
        return known[status]
    return _normalized_code(provider_code) or (
        "invalid_request" if status is not None else "provider_unavailable"
    )


def _limit_classification(
    status: int | None,
    reason_code: str,
    provider_code: str | None,
    error_message: str | None,
    metadata_error_type: str | None,
) -> str | None:
    text = " ".join(
        value.lower()
        for value in (provider_code, error_message, metadata_error_type)
        if value
    )
    if reason_code in {"authentication_failed", "invalid_request", "model_unavailable"}:
        return ACCOUNT_CONFIGURATION_ERROR
    if reason_code in {"context_limit_exceeded", "request_too_large"}:
        return REQUEST_TOKEN_LIMIT_EXCEEDED
    if "concurrent" in text:
        return CONCURRENT_LIMIT
    if reason_code == "quota_exhausted":
        return DAILY_TOKEN_QUOTA_EXHAUSTED
    if status == 429:
        if reason_code in {"rate_limited", "token_quota_exhausted"}:
            return RATE_LIMITED_TEMPORARY
        return UNKNOWN_PROVIDER_LIMIT
    return None


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


def _sanitized_error_body(response: httpx.Response) -> str | None:
    """Return bounded provider error diagnostics without echoing request content."""

    code, message, error_type, provider_code = _error_fields(response)
    if any((code, message, error_type, provider_code)):
        return json.dumps(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "error_type": error_type,
                    "provider_code": provider_code,
                }
            },
            separators=(",", ":"),
        )
    content_type = _safe_text(response.headers.get("content-type"), limit=128)
    return (
        f"provider_error_without_safe_fields body_bytes={len(response.content)} "
        f"content_type={content_type or 'unknown'}"
    )


def _network_categories(exc: httpx.RequestError) -> tuple[str | None, str]:
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout", "timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout", "timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout", "timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout", "timeout"
    text = str(exc).lower()
    if "name or service not known" in text or "getaddrinfo" in text or "dns" in text:
        return None, "dns"
    if "ssl" in text or "tls" in text or "certificate" in text:
        return None, "tls"
    if isinstance(exc, httpx.ConnectError):
        return None, "connection"
    return None, "network"


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
    finish_reason: str | None = None
    response_size_bytes: int | None = None
    response_character_count: int | None = None
    extraction_note: str | None = None
    prompt_character_count: int | None = None


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
        trigger: str | None = None,
        idempotency_key: str | None = None,
        time_bucket: str | None = None,
        attempt: int = 1,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        attempt_type: str | None = None,
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


def _provider_token_usage(envelope: object) -> dict[str, int] | None:
    if not isinstance(envelope, dict):
        return None
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return None
    token_usage = {
        key: int(value)
        for key, source in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        )
        if isinstance((value := usage.get(source)), (int, float))
    }
    return token_usage or None


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
        probe_context = {
            "provider": self.provider,
            "request_kind": "model_probe",
            "endpoint_host": urlsplit(endpoint).netloc,
            "endpoint_path": urlsplit(endpoint).path,
        }
        logger.info(
            "ai_provider.model_probe.started provider=%s",
            self.provider,
            extra=probe_context,
        )
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
        except httpx.HTTPStatusError as exc:
            code, message, error_type, provider_code = _error_fields(exc.response)
            logger.warning(
                "ai_provider.model_probe.failed provider=%s status=%s",
                self.provider,
                exc.response.status_code,
                extra={
                    **probe_context,
                    "provider": self.provider,
                    "endpoint": endpoint,
                    "status_code": exc.response.status_code,
                    "sanitized_error_code": _reason_code(
                        exc.response.status_code,
                        code,
                        message,
                        error_type,
                    ),
                    "provider_error_code": code,
                    "provider_error_message": message,
                    "provider_error_type": error_type,
                    "provider_code": provider_code,
                    "exception_type": type(exc).__name__,
                },
            )
            return ()
        except httpx.RequestError as exc:
            logger.warning(
                "ai_provider.model_probe.failed provider=%s status=network_error",
                self.provider,
                extra={
                    **probe_context,
                    "provider": self.provider,
                    "endpoint": endpoint,
                    "status_code": None,
                    "sanitized_error_code": "provider_unavailable",
                    "exception_type": type(exc).__name__,
                },
            )
            return ()
        except (ValueError, AttributeError) as exc:
            logger.warning(
                "ai_provider.model_probe.failed provider=%s status=decoding_error",
                self.provider,
                extra={
                    **probe_context,
                    "provider": self.provider,
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "sanitized_error_code": "response_decoding_failed",
                    "exception_type": type(exc).__name__,
                },
            )
            return ()
        models = tuple(
            str(item["id"])
            for item in data
            if isinstance(item, dict) and item.get("id")
        )
        logger.info(
            "ai_provider.model_probe.completed provider=%s status=200 models=%s",
            self.provider,
            len(models),
            extra={
                **probe_context,
                "status_code": 200,
                "model_count": len(models),
            },
        )
        return models

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
        trigger: str | None = None,
        idempotency_key: str | None = None,
        time_bucket: str | None = None,
        attempt: int = 1,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        attempt_type: str | None = None,
    ) -> AIProviderCompletion:
        if not self.api_key:
            raise ConfigurationError(
                f"TEN_{self.provider.upper()}_API_KEY is required for {self.provider}"
            )
        endpoint = f"{self.base_url}/chat/completions"
        endpoint_parts = urlsplit(endpoint)
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
            "trigger": trigger,
            "idempotency_key": idempotency_key,
            "time_bucket": time_bucket,
            "attempt": attempt,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "request_id": request_id,
            "endpoint": endpoint,
            "endpoint_host": endpoint_parts.netloc,
            "endpoint_path": endpoint_parts.path,
            "http_method": "POST",
            "status_code": None,
            "error_type": None,
            "sanitized_error_code": None,
            "latency_ms": None,
            "provider_request_id": None,
            "rate_limit_remaining": None,
            "rate_limit_reset": None,
            "exception_type": None,
            "network_error_category": None,
            "timeout_category": None,
            "attempt_type": attempt_type,
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
                serialized_request_bytes=metrics.serialized_request_bytes,
                estimated_input_tokens=metrics.estimated_input_tokens,
                attempt_type=attempt_type,
            ) from exc
        except httpx.RequestError as exc:
            timeout_category, network_category = _network_categories(exc)
            details = AIProviderFailureDetails(
                provider=self.provider,
                reason_code=(
                    "request_timeout"
                    if timeout_category is not None
                    else "provider_unavailable"
                ),
                phase="http_request",
                endpoint=endpoint,
                model=model,
                endpoint_host=endpoint_parts.netloc,
                endpoint_path=endpoint_parts.path,
                request_id=request_id,
                cycle_id=cycle_id,
                error_message=_safe_text(str(exc)),
                elapsed_ms=(perf_counter() - started) * 1000,
                exception_class=type(exc).__name__,
                serialized_request_bytes=metrics.serialized_request_bytes,
                estimated_input_tokens=metrics.estimated_input_tokens,
                timeout_category=timeout_category,
                network_error_category=network_category,
                attempt_type=attempt_type,
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
        raw_content: object = None
        finish_reason: str | None = None
        extraction_note: str | None = None
        token_usage: dict[str, int] | None = None
        try:
            envelope = response.json()
            token_usage = _provider_token_usage(envelope)
            choice = envelope["choices"][0]
            raw_content = choice["message"]["content"]
            finish_reason = _safe_text(choice.get("finish_reason"), limit=64)
            parsed, extraction_note = extract_single_json_object(raw_content)
        except (KeyError, IndexError, json.JSONDecodeError, TypeError, ValueError) as exc:
            schema_error_code = (
                exc.reason_code
                if isinstance(exc, ProviderJSONDecodeError)
                else "response_envelope_invalid"
            )
            if finish_reason == "length":
                schema_error_code = "finish_reason_length"
            details = AIProviderFailureDetails(
                provider=self.provider,
                reason_code=(
                    "truncated_response"
                    if schema_error_code
                    in {"truncated_response", "finish_reason_length"}
                    else "response_decoding_failed"
                ),
                phase="response_decoding",
                endpoint=endpoint,
                model=model,
                endpoint_host=endpoint_parts.netloc,
                endpoint_path=endpoint_parts.path,
                request_id=request_id,
                cycle_id=cycle_id,
                http_status=response.status_code,
                content_type=_safe_text(headers.get("content-type"), limit=128),
                body_length=len(response.content),
                sanitized_response_body=_sanitized_error_body(response),
                serialized_request_bytes=metrics.serialized_request_bytes,
                estimated_input_tokens=metrics.estimated_input_tokens,
                attempt_type=attempt_type,
                provider_request_id=provider_request_id,
                elapsed_ms=elapsed,
                exception_class=type(exc).__name__,
                finish_reason=finish_reason,
                response_size_bytes=len(response.content),
                response_character_count=(
                    len(raw_content) if isinstance(raw_content, str) else None
                ),
                provider_input_tokens=(
                    token_usage.get("input_tokens") if token_usage else None
                ),
                provider_output_tokens=(
                    token_usage.get("output_tokens") if token_usage else None
                ),
                provider_total_tokens=(
                    token_usage.get("total_tokens") if token_usage else None
                ),
                schema_error_code=schema_error_code,
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
                endpoint_host=endpoint_parts.netloc,
                endpoint_path=endpoint_parts.path,
                request_id=request_id,
                cycle_id=cycle_id,
                http_status=response.status_code,
                body_length=len(response.content),
                sanitized_response_body=_sanitized_error_body(response),
                serialized_request_bytes=metrics.serialized_request_bytes,
                estimated_input_tokens=metrics.estimated_input_tokens,
                attempt_type=attempt_type,
                provider_request_id=provider_request_id,
                elapsed_ms=elapsed,
                exception_class="TypeError",
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            self._log_failure(details, instrument=instrument, ums_boundary=ums_boundary, attempt=attempt)
            raise AIProviderRequestError(details)
        raw_json_text = (
            raw_content
            if isinstance(raw_content, str)
            else json.dumps(raw_content, ensure_ascii=False, separators=(",", ":"))
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
            raw_json_text=raw_json_text,
            finish_reason=finish_reason,
            response_size_bytes=len(response.content),
            response_character_count=len(raw_json_text),
            extraction_note=extraction_note,
            prompt_character_count=(
                metrics.system_prompt_characters + metrics.user_prompt_characters
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
        serialized_request_bytes: int,
        estimated_input_tokens: int,
        attempt_type: str | None,
    ) -> AIProviderRequestError:
        error_code, error_message, metadata_error_type, metadata_provider_code = (
            _error_fields(response)
        )
        headers = response.headers
        endpoint = f"{self.base_url}/chat/completions"
        endpoint_parts = urlsplit(endpoint)
        request_limit = _safe_text(headers.get("x-ratelimit-limit-requests"))
        request_remaining = _safe_text(headers.get("x-ratelimit-remaining-requests"))
        request_reset = _safe_text(headers.get("x-ratelimit-reset-requests"))
        token_limit = _safe_text(headers.get("x-ratelimit-limit-tokens"))
        token_remaining = _safe_text(headers.get("x-ratelimit-remaining-tokens"))
        token_reset = _safe_text(headers.get("x-ratelimit-reset-tokens"))
        reason_code = _reason_code(
            response.status_code,
            error_code,
            error_message,
            metadata_error_type,
        )
        details = AIProviderFailureDetails(
            provider=self.provider,
            reason_code=reason_code,
            phase="http_request",
            endpoint=endpoint,
            model=model,
            endpoint_host=endpoint_parts.netloc,
            endpoint_path=endpoint_parts.path,
            request_id=request_id,
            cycle_id=cycle_id,
            http_status=response.status_code,
            error_code=error_code,
            error_message=error_message,
            metadata_error_type=metadata_error_type,
            metadata_provider_code=metadata_provider_code,
            content_type=_safe_text(headers.get("content-type"), limit=128),
            body_length=len(response.content),
            sanitized_response_body=_sanitized_error_body(response),
            serialized_request_bytes=serialized_request_bytes,
            estimated_input_tokens=estimated_input_tokens,
            attempt_type=attempt_type,
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
            limit_classification=_limit_classification(
                response.status_code,
                reason_code,
                error_code,
                error_message,
                metadata_error_type,
            ),
            response_size_bytes=len(response.content),
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
            "ai_provider.request.failure_diagnostic",
            extra={
                "provider": details.provider,
                "request_id": details.request_id,
                "model": details.model,
                "endpoint_host": details.endpoint_host,
                "endpoint_path": details.endpoint_path,
                "http_method": details.http_method,
                "instrument": instrument,
                "cycle_id": details.cycle_id,
                "ums_boundary": ums_boundary,
                "attempt": attempt,
                "fallback_used": details.fallback_used,
                "fallback_reason": details.fallback_reason,
                "status_code": details.http_status,
                "error_type": details.metadata_error_type,
                "provider_error_code": details.error_code,
                "provider_error_type": details.metadata_error_type,
                "provider_response": details.sanitized_response_body,
                "sanitized_error_code": details.reason_code,
                "latency_ms": details.elapsed_ms,
                "provider_request_id": details.provider_request_id,
                "limit_classification": details.limit_classification,
                "rate_limit_remaining": details.rate_limit_remaining,
                "rate_limit_reset": details.rate_limit_reset,
                "rate_limit_request_limit": details.rate_limit_request_limit,
                "rate_limit_request_remaining": details.rate_limit_request_remaining,
                "rate_limit_request_reset": details.rate_limit_request_reset,
                "rate_limit_token_limit": details.rate_limit_token_limit,
                "rate_limit_token_remaining": details.rate_limit_token_remaining,
                "rate_limit_token_reset": details.rate_limit_token_reset,
                "provider_input_tokens": details.provider_input_tokens,
                "provider_output_tokens": details.provider_output_tokens,
                "provider_total_tokens": details.provider_total_tokens,
                "exception_type": details.exception_class,
                "safe_exception_message": details.error_message,
                "network_error_category": (
                    details.network_error_category
                ),
                "timeout_category": details.timeout_category,
                "serialized_request_bytes": details.serialized_request_bytes,
                "estimated_input_tokens": details.estimated_input_tokens,
                "response_content_type": details.content_type,
                "response_body_length": details.body_length,
                "response_size_bytes": details.response_size_bytes,
                "finish_reason": details.finish_reason,
                "schema_error_code": details.schema_error_code,
                "schema_error_path": details.schema_error_path,
                "attempt_type": details.attempt_type,
            },
        )
