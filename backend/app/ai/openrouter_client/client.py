"""Narrow OpenRouter API client with explicit JSON output handling."""

import json
import logging
from abc import ABC, abstractmethod
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


def _reason_code(status: int | None, provider_code: str | None) -> str:
    known = {
        400: "openrouter_invalid_request",
        401: "openrouter_authentication_failed",
        402: "openrouter_insufficient_credits",
        403: "openrouter_permission_denied",
        413: "openrouter_payload_too_large",
        429: "openrouter_rate_limited",
        502: "openrouter_provider_unavailable",
        503: "openrouter_provider_unavailable",
    }
    if status in known:
        return known[status]
    if provider_code:
        normalized = "".join(character if character.isalnum() else "_" for character in provider_code.lower()).strip("_")
        if normalized:
            return f"openrouter_{normalized}"[:96]
    return "openrouter_http_error" if status is not None else "openrouter_transport_error"


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
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(payload)}],
        }
        started = perf_counter()
        logger.info(
            "openrouter.request.started",
            extra={
                "request_id": request_id,
                "cycle_id": cycle_id,
                "model": model,
                "endpoint": endpoint,
                "failure_phase": None,
            },
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.post(endpoint, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=request)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_code, error_message, metadata_error_type, metadata_provider_code = _error_fields(exc.response)
            details = OpenRouterFailureDetails(
                reason_code=_reason_code(exc.response.status_code, error_code),
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
                reason_code="openrouter_transport_error",
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
                "response_content_type": _safe_text(response.headers.get("content-type"), limit=128),
                "response_body_length": len(response.content),
                "retry_after": _safe_text(response.headers.get("retry-after"), limit=128),
                "elapsed_ms": (perf_counter() - started) * 1000,
                "failure_phase": None,
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

