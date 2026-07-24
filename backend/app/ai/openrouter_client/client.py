"""Narrow OpenRouter API client with explicit JSON output handling."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Any

import httpx

from backend.app.core.exceptions import (
    ConfigurationError,
    OpenRouterFailureDetails,
    OpenRouterRequestError,
)

logger = logging.getLogger(__name__)

_FAILURE_PHASES = (
    "request_validation",
    "http_request",
    "response_decoding",
    "structured_output_validation",
    "domain_parsing",
    "persistence",
)


def _safe_text(value: object, *, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] or None


def _reason_code(
    status: int | None,
    provider_code: str | None,
    error_message: str | None = None,
    metadata_error_type: str | None = None,
) -> str:
    error_text = " ".join(
        value.lower()
        for value in (provider_code, error_message, metadata_error_type)
        if value
    )
    if "context" in error_text and ("limit" in error_text or "length" in error_text):
        return "context_limit_exceeded"
    if "prompt token" in error_text and "limit" in error_text:
        return "key_limit_exhausted"
    if "maximum cost" in error_text or "max cost" in error_text:
        return "maximum_cost_exceeded"
    if "no eligible provider" in error_text or "no provider" in error_text:
        return "no_eligible_provider"
    if status is not None and 500 <= status <= 599:
        return "provider_unavailable"
    known = {
        400: "invalid_request",
        401: "authentication_failed",
        402: "payment_blocked",
        403: "authentication_failed",
        413: "request_too_large",
        429: "rate_limited",
        502: "provider_unavailable",
        503: "provider_unavailable",
    }
    if status in known:
        return known[status]
    if provider_code:
        normalized = "".join(character if character.isalnum() else "_" for character in provider_code.lower()).strip("_")
        if normalized:
            return normalized[:96]
    return "invalid_request" if status is not None else "provider_unavailable"


def _error_fields(response: httpx.Response) -> tuple[str | None, str | None, str | None, str | None]:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None, None, None, None
    if not isinstance(body, dict):
        return None, None, None, None
    error = body.get("error")
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


def _failure_flags(phase: str) -> dict[str, bool]:
    return {f"failed_during_{candidate}": phase == candidate for candidate in _FAILURE_PHASES}


@dataclass(frozen=True)
class OpenRouterRequestMetrics:
    serialized_request_bytes: int
    estimated_input_tokens: int
    maximum_output_tokens: int
    estimated_maximum_cost_usd: float
    system_prompt_characters: int
    user_prompt_characters: int
    message_count: int
    tool_definition_bytes: int = 0
    response_schema_bytes: int = 0


def build_request_body(
    *,
    system_prompt: str,
    payload: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the single canonical request body used by measurement and HTTP."""

    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
    }


def measure_request_body(
    body: dict[str, Any],
    *,
    input_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
) -> OpenRouterRequestMetrics:
    """Return conservative, content-free request size/token/cost measurements.

    Llama's exact tokenizer is not an application dependency.  Three UTF-8 bytes per
    token deliberately overestimates the measured production request (48,161 provider
    tokens for 211,104 bytes) and is therefore safe for a hard preflight guard.
    """

    encoded = httpx.Request("POST", "https://openrouter.invalid", json=body).content
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
    return OpenRouterRequestMetrics(
        serialized_request_bytes=len(encoded),
        estimated_input_tokens=estimated_input,
        maximum_output_tokens=maximum_output,
        estimated_maximum_cost_usd=maximum_cost,
        system_prompt_characters=system_chars,
        user_prompt_characters=user_chars,
        message_count=len(messages),
        response_schema_bytes=len(
            json.dumps(
                body.get("response_format", {}),
                separators=(",", ":"),
            ).encode()
        ),
    )


def _log_failure(details: OpenRouterFailureDetails) -> None:
    logger.error(
        "openrouter.request.failed",
        extra={
            "request_id": details.request_id,
            "cycle_id": details.cycle_id,
            "model": details.model,
            "endpoint": details.endpoint,
            "http_status": details.http_status,
            "openrouter_error_code": details.error_code,
            "openrouter_error_message": details.error_message,
            "metadata_error_type": details.metadata_error_type,
            "metadata_provider_code": details.metadata_provider_code,
            "response_content_type": details.content_type,
            "response_body_length": details.body_length,
            "retry_after": details.retry_after,
            "elapsed_ms": details.elapsed_ms,
            "exception_class": details.exception_class,
            "failure_phase": details.phase,
            "failure_reason_code": details.reason_code,
            **_failure_flags(details.phase),
        },
    )


class OpenRouterClient(ABC):
    @abstractmethod
    async def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        model: str,
        temperature: float,
        max_tokens: int,
        request_id: str | None = None,
        cycle_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a decoded JSON object from a selected OpenRouter model."""


class HttpOpenRouterClient(OpenRouterClient):
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float = 30.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        model: str,
        temperature: float,
        max_tokens: int,
        request_id: str | None = None,
        cycle_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ConfigurationError("TEN_OPENROUTER_API_KEY is required for AI scoring")
        endpoint = f"{self.base_url}/chat/completions"
        request = {
            **build_request_body(
                system_prompt=system_prompt,
                payload=payload,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        }
        request_metrics = measure_request_body(
            request,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )
        analysis_context = payload.get("analysis_context")
        response_contract = payload.get("response_contract")
        request_schema_version = (
            analysis_context.get("schema_version")
            if isinstance(analysis_context, dict)
            else None
        )
        response_schema_version = (
            response_contract.get("schema_version")
            if isinstance(response_contract, dict)
            else None
        )
        started = perf_counter()
        logger.info(
            "openrouter.request.started",
            extra={
                "request_id": request_id,
                "cycle_id": cycle_id,
                "model": model,
                "endpoint": endpoint,
                "request_schema_version": request_schema_version,
                "response_schema_version": response_schema_version,
                "failure_phase": None,
                "serialized_request_bytes": request_metrics.serialized_request_bytes,
                "estimated_input_tokens": request_metrics.estimated_input_tokens,
                "maximum_output_tokens": request_metrics.maximum_output_tokens,
                "message_count": request_metrics.message_count,
                "system_prompt_characters": request_metrics.system_prompt_characters,
                "user_prompt_characters": request_metrics.user_prompt_characters,
                "tool_definition_bytes": request_metrics.tool_definition_bytes,
                "response_schema_bytes": request_metrics.response_schema_bytes,
            },
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.post(endpoint, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=request)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_code, error_message, metadata_error_type, metadata_provider_code = _error_fields(exc.response)
            details = OpenRouterFailureDetails(
                reason_code=_reason_code(
                    exc.response.status_code,
                    error_code,
                    error_message,
                    metadata_error_type,
                ),
                phase="http_request",
                request_id=request_id,
                cycle_id=cycle_id,
                model=model,
                endpoint=endpoint,
                http_status=exc.response.status_code,
                error_code=error_code,
                error_message=error_message,
                metadata_error_type=metadata_error_type,
                metadata_provider_code=metadata_provider_code,
                content_type=_safe_text(exc.response.headers.get("content-type"), limit=128),
                body_length=len(exc.response.content),
                retry_after=_safe_text(exc.response.headers.get("retry-after"), limit=128),
                elapsed_ms=(perf_counter() - started) * 1000,
                exception_class=type(exc).__name__,
            )
            _log_failure(details)
            raise OpenRouterRequestError(details) from exc
        except httpx.RequestError as exc:
            details = OpenRouterFailureDetails(
                reason_code="provider_unavailable",
                phase="http_request",
                request_id=request_id,
                cycle_id=cycle_id,
                model=model,
                endpoint=endpoint,
                elapsed_ms=(perf_counter() - started) * 1000,
                exception_class=type(exc).__name__,
            )
            _log_failure(details)
            raise OpenRouterRequestError(details) from exc
        logger.info(
            "openrouter.request.http_completed",
            extra={
                "request_id": request_id,
                "cycle_id": cycle_id,
                "model": model,
                "endpoint": endpoint,
                "http_status": response.status_code,
                "request_schema_version": request_schema_version,
                "response_schema_version": response_schema_version,
                "response_content_type": _safe_text(response.headers.get("content-type"), limit=128),
                "response_body_length": len(response.content),
                "retry_after": _safe_text(response.headers.get("retry-after"), limit=128),
                "elapsed_ms": (perf_counter() - started) * 1000,
                "failure_phase": None,
            },
        )
        logger.info(
            "openrouter.response.received",
            extra={
                "request_id": request_id,
                "cycle_id": cycle_id,
                "model": model,
                "endpoint": endpoint,
                "http_status": response.status_code,
                "request_schema_version": request_schema_version,
                "response_schema_version": response_schema_version,
                "response_content_type": _safe_text(response.headers.get("content-type"), limit=128),
                "response_body_length": len(response.content),
                "elapsed_ms": (perf_counter() - started) * 1000,
            },
        )
        try:
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            details = OpenRouterFailureDetails(
                reason_code="openrouter_response_decoding_failed",
                phase="response_decoding",
                request_id=request_id,
                cycle_id=cycle_id,
                model=model,
                endpoint=endpoint,
                http_status=response.status_code,
                content_type=_safe_text(response.headers.get("content-type"), limit=128),
                body_length=len(response.content),
                retry_after=_safe_text(response.headers.get("retry-after"), limit=128),
                elapsed_ms=(perf_counter() - started) * 1000,
                exception_class=type(exc).__name__,
            )
            _log_failure(details)
            raise OpenRouterRequestError(details) from exc
        if not isinstance(parsed, dict):
            details = OpenRouterFailureDetails(
                reason_code="openrouter_domain_parsing_failed",
                phase="domain_parsing",
                request_id=request_id,
                cycle_id=cycle_id,
                model=model,
                endpoint=endpoint,
                http_status=response.status_code,
                content_type=_safe_text(response.headers.get("content-type"), limit=128),
                body_length=len(response.content),
                elapsed_ms=(perf_counter() - started) * 1000,
                exception_class="TypeError",
            )
            _log_failure(details)
            raise OpenRouterRequestError(details)
        return parsed

