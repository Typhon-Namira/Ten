"""Turns a grounded `ExplainabilityContext` into natural-language explanation — the only place in
this module that talks to an LLM, and the only place a request can fail (a bad/unreachable
OpenRouter call degrades to a clear error, never a fabricated answer standing in for a real one).
"""

from __future__ import annotations

from datetime import UTC
from hashlib import sha256
import json
import logging
from typing import Any

from pydantic import ValidationError

from backend.app.ai.openrouter_client.client import OpenRouterClient
from backend.app.ai.prompts.loader import PromptLoader

from .models import ChatTurn, Explanation, ExplainabilityContext

logger = logging.getLogger(__name__)

PROMPT_VERSION = "explain_v1"


class ExplainabilityService:
    def __init__(self, client: OpenRouterClient, prompts: PromptLoader, *, model: str, temperature: float = 0.2, max_tokens: int = 1200) -> None:
        self.client = client
        self.prompts = prompts
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _complete(
        self,
        context: ExplainabilityContext,
        payload: dict[str, Any],
        *,
        trigger: str,
    ) -> dict[str, Any]:
        generated_at = context.generated_at.astimezone(UTC)
        bucket = generated_at.replace(
            minute=(generated_at.minute // 10) * 10,
            second=0,
            microsecond=0,
        )
        idempotency_key = sha256(
            json.dumps(
                {
                    "trigger": trigger,
                    "instrument": context.instrument,
                    "time_bucket": bucket.isoformat(),
                    "payload": payload,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        call_context = {
            "trigger": trigger,
            "instrument": context.instrument,
            "idempotency_key": idempotency_key,
            "ten_minute_bucket": bucket.isoformat(),
        }
        logger.info("explainability.provider_call.started", extra=call_context)
        try:
            raw = await self.client.complete_json(
                system_prompt=self.prompts.load(PROMPT_VERSION),
                payload=payload,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.info(
                "explainability.provider_call.completed",
                extra={
                    **call_context,
                    "result": "failed",
                    "exception_class": type(exc).__name__,
                },
            )
            raise
        logger.info(
            "explainability.provider_call.completed",
            extra={**call_context, "result": "success"},
        )
        return raw

    async def explain(self, context: ExplainabilityContext) -> Explanation:
        """The one call every `/explain/*` endpoint routes through. Never raises for a bad model
        response — an explanation that fails validation is a *degraded* result the caller renders
        as `status: "error"` (matching the rest of TEN's never-500 observability policy), not an
        exception that turns into an opaque 500."""
        payload = context.model_dump(mode="json")
        raw = await self._complete(
            context,
            payload,
            trigger="explainability_api",
        )
        try:
            return Explanation.model_validate(raw)
        except ValidationError as exc:
            logger.warning("explainability.invalid_model_response", extra={"errors": exc.errors()})
            raise

    async def chat(self, context: ExplainabilityContext, history: tuple[ChatTurn, ...]) -> Explanation:
        """Conversation history travels *inside* the grounded JSON payload (as one more field the
        model reads), rather than as separate chat-API message turns — `OpenRouterClient` only
        exposes one system+user exchange per call, and folding history into the same grounded
        blob keeps the "one JSON in, one JSON out" contract identical to every other explanation."""
        payload = context.model_dump(mode="json")
        payload["conversation_history"] = [turn.model_dump(mode="json") for turn in history]
        raw = await self._complete(
            context,
            payload,
            trigger="explainability_chat_api",
        )
        return Explanation.model_validate(raw)
