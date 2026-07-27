"""Direct, fallback-free Cerebras diagnostics for production isolation.

Run inside the configured service container:

    python -m backend.app.ai_reasoning.cerebras_probe text
    python -m backend.app.ai_reasoning.cerebras_probe json
    python -m backend.app.ai_reasoning.cerebras_probe analysis --request-json request.json

The analysis file must contain one serialized ``AIReasoningRequest``. Output is
bounded metadata only; keys, prompts, payloads, and full model output are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from backend.app.ai.provider_client import HttpAIProviderClient
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.config import YamlConfigRepository
from backend.app.core.config.settings import Settings
from backend.app.core.exceptions import AIProviderRequestError, ConfigurationError

from .config import AIReasoningConfig
from .models import AIReasoningRequest
from .provider import CerebrasProvider
from .setup_families import SetupFamilyRegistry
from .validation import StructuredAIOutputError, StructuredAIOutputValidator

ProbeMode = Literal["text", "json", "analysis"]


@dataclass(frozen=True)
class CerebrasProbeResult:
    mode: ProbeMode
    base_url: str
    endpoint_host: str
    endpoint_path: str
    model: str
    compatibility_mode: str
    api_key_present: bool
    success: bool
    http_status: int | None
    provider_request_id: str | None
    content_type: str | None
    rate_limit_remaining: str | None
    rate_limit_reset: str | None
    elapsed_ms: float
    sanitized_response: str
    validation_passed: bool | None = None


def _safe_response(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"non_json_response_bytes={len(response.content)}"
    if response.is_error:
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            return json.dumps(
                {
                    "error": {
                        "type": error.get("type"),
                        "code": error.get("code"),
                        "message": str(error.get("message", ""))[:500],
                    }
                },
                separators=(",", ":"),
            )
        return f"provider_error_json_type={type(body).__name__}"
    content = (
        body.get("choices", [{}])[0].get("message", {}).get("content")
        if isinstance(body, dict)
        else None
    )
    return f"completion_received content_characters={len(str(content or ''))}"


def _completion_content(response: httpx.Response) -> str | None:
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


async def run_minimal_probe(
    mode: Literal["text", "json"],
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CerebrasProbeResult:
    endpoint = f"{settings.cerebras_base_url.rstrip('/')}/chat/completions"
    parts = urlsplit(endpoint)
    body: dict[str, Any] = {
        "model": settings.cerebras_model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return the word OK."
                    if mode == "text"
                    else 'Return exactly this JSON object: {"status":"ok"}'
                ),
            }
        ],
        "max_tokens": 20,
        "temperature": 0,
    }
    if mode == "json":
        body["response_format"] = {"type": "json_object"}
    started = perf_counter()
    response: httpx.Response | None = None
    try:
        if not settings.cerebras_api_key:
            raise RuntimeError("TEN_CEREBRAS_API_KEY is not configured")
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds,
            transport=transport,
        ) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {settings.cerebras_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        content = _completion_content(response)
        if mode == "text":
            output_valid = content is not None and content.strip() == "OK"
        else:
            try:
                decoded = json.loads(content or "")
            except json.JSONDecodeError:
                output_valid = False
            else:
                output_valid = decoded == {"status": "ok"}
        success = response.is_success and output_valid
        sanitized = _safe_response(response)
        if response.is_success and not output_valid:
            sanitized = f"probe_output_invalid {_safe_response(response)}"
    except httpx.RequestError as exc:
        success = False
        sanitized = f"network_error={type(exc).__name__}"
    except RuntimeError as exc:
        success = False
        sanitized = str(exc)
    headers = response.headers if response is not None else {}
    return CerebrasProbeResult(
        mode=mode,
        base_url=settings.cerebras_base_url,
        endpoint_host=parts.netloc,
        endpoint_path=parts.path,
        model=settings.cerebras_model,
        compatibility_mode="openai_chat_completions",
        api_key_present=bool(settings.cerebras_api_key),
        success=success,
        http_status=response.status_code if response is not None else None,
        provider_request_id=headers.get("x-request-id") or headers.get("request-id"),
        content_type=headers.get("content-type"),
        rate_limit_remaining=(
            headers.get("x-ratelimit-remaining-requests")
            or headers.get("x-ratelimit-remaining")
        ),
        rate_limit_reset=(
            headers.get("x-ratelimit-reset-requests")
            or headers.get("x-ratelimit-reset")
        ),
        elapsed_ms=(perf_counter() - started) * 1000,
        sanitized_response=sanitized,
    )


async def run_analysis_probe(
    request: AIReasoningRequest,
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CerebrasProbeResult:
    configs = YamlConfigRepository()
    config = configs.load_model("ai_reasoning", AIReasoningConfig)
    registry = SetupFamilyRegistry.from_yaml(configs)
    client = HttpAIProviderClient(
        "cerebras",
        settings.cerebras_api_key,
        settings.cerebras_base_url,
        settings.request_timeout_seconds,
        transport=transport,
    )
    provider = CerebrasProvider(
        client,
        PromptLoader(Path(__file__).resolve().parent / "prompts"),
        model=settings.cerebras_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        target_input_tokens=config.target_input_tokens,
        warning_input_tokens=config.warning_input_tokens,
        hard_input_tokens=config.hard_input_tokens,
        absolute_max_output_tokens=config.absolute_max_output_tokens,
        maximum_request_cost_usd=config.maximum_request_cost_usd,
        input_cost_per_million_usd=config.input_cost_per_million_usd,
        output_cost_per_million_usd=config.output_cost_per_million_usd,
        setup_family_ids=tuple(item.setup_family_id for item in registry.all()),
    )
    endpoint = f"{settings.cerebras_base_url.rstrip('/')}/chat/completions"
    parts = urlsplit(endpoint)
    started = perf_counter()
    try:
        response = await provider.reason(
            request,
            prompt_version=request.prompt_version,
        )
        StructuredAIOutputValidator().validate_analysis(response.raw_output)
        metadata = response.operational_metadata or {}
        status_code = metadata.get("status_code")
        return CerebrasProbeResult(
            mode="analysis",
            base_url=settings.cerebras_base_url,
            endpoint_host=parts.netloc,
            endpoint_path=parts.path,
            model=settings.cerebras_model,
            compatibility_mode="openai_chat_completions_strict_json_schema",
            api_key_present=bool(settings.cerebras_api_key),
            success=True,
            http_status=int(status_code) if isinstance(status_code, (int, str)) else 200,
            provider_request_id=(
                str(metadata["provider_request_id"])
                if metadata.get("provider_request_id")
                else None
            ),
            content_type="application/json",
            rate_limit_remaining=(
                str(metadata["rate_limit_remaining"])
                if metadata.get("rate_limit_remaining")
                else None
            ),
            rate_limit_reset=(
                str(metadata["rate_limit_reset"])
                if metadata.get("rate_limit_reset")
                else None
            ),
            elapsed_ms=(perf_counter() - started) * 1000,
            sanitized_response="validated_ai_analysis_received",
            validation_passed=True,
        )
    except AIProviderRequestError as exc:
        details = exc.details
        return CerebrasProbeResult(
            mode="analysis",
            base_url=settings.cerebras_base_url,
            endpoint_host=parts.netloc,
            endpoint_path=parts.path,
            model=settings.cerebras_model,
            compatibility_mode="openai_chat_completions_strict_json_schema",
            api_key_present=bool(settings.cerebras_api_key),
            success=False,
            http_status=details.http_status,
            provider_request_id=details.provider_request_id,
            content_type=details.content_type,
            rate_limit_remaining=details.rate_limit_remaining,
            rate_limit_reset=details.rate_limit_reset,
            elapsed_ms=(perf_counter() - started) * 1000,
            sanitized_response=(
                details.sanitized_response_body or details.reason_code
            ),
            validation_passed=False,
        )
    except StructuredAIOutputError as exc:
        return CerebrasProbeResult(
            mode="analysis",
            base_url=settings.cerebras_base_url,
            endpoint_host=parts.netloc,
            endpoint_path=parts.path,
            model=settings.cerebras_model,
            compatibility_mode="openai_chat_completions_strict_json_schema",
            api_key_present=bool(settings.cerebras_api_key),
            success=False,
            http_status=200,
            provider_request_id=None,
            content_type="application/json",
            rate_limit_remaining=None,
            rate_limit_reset=None,
            elapsed_ms=(perf_counter() - started) * 1000,
            sanitized_response=f"structured_validation_failed:{exc.errors[0]}",
            validation_passed=False,
        )
    except ConfigurationError as exc:
        return CerebrasProbeResult(
            mode="analysis",
            base_url=settings.cerebras_base_url,
            endpoint_host=parts.netloc,
            endpoint_path=parts.path,
            model=settings.cerebras_model,
            compatibility_mode="openai_chat_completions_strict_json_schema",
            api_key_present=False,
            success=False,
            http_status=None,
            provider_request_id=None,
            content_type=None,
            rate_limit_remaining=None,
            rate_limit_reset=None,
            elapsed_ms=(perf_counter() - started) * 1000,
            sanitized_response=str(exc),
            validation_passed=False,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one direct Cerebras diagnostic")
    parser.add_argument("mode", choices=("text", "json", "analysis"))
    parser.add_argument("--request-json", type=Path)
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    settings = Settings()
    if args.mode == "analysis":
        if args.request_json is None:
            raise SystemExit("--request-json is required for analysis mode")
        request = AIReasoningRequest.model_validate_json(
            args.request_json.read_text(encoding="utf-8")
        )
        result = await run_analysis_probe(request, settings)
    else:
        result = await run_minimal_probe(args.mode, settings)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
