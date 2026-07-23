"""Deterministic point-in-time replay and recorded LLM-response reuse."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from typing import Any, Protocol

from backend.app.market_state import UnifiedMarketState

from .models import ReplayLLMMode


def replay_request_hash(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class RecordedLLMResponse:
    request_hash: str
    prompt_version: str
    model_identifier: str
    temperature: float
    generation_parameters: dict[str, Any]
    structured_response: dict[str, Any]


class ReplayResponseStore(Protocol):
    async def get(self, request_hash: str) -> RecordedLLMResponse | None: ...


class InMemoryReplayResponseStore:
    def __init__(self, responses: tuple[RecordedLLMResponse, ...] = ()) -> None:
        self.responses = {item.request_hash: item for item in responses}

    async def get(self, request_hash: str) -> RecordedLLMResponse | None:
        return self.responses.get(request_hash)


class DeterministicReplayAdapter:
    """Never contacts a live model in recorded or deterministic-baseline replay."""

    def __init__(self, store: ReplayResponseStore, *, baseline: dict[str, Any] | None = None) -> None:
        self.store = store
        self.baseline = baseline

    async def response(self, request: dict[str, Any], mode: ReplayLLMMode) -> dict[str, Any]:
        request_hash = replay_request_hash(request)
        if mode == ReplayLLMMode.RECORDED_RESPONSE:
            recorded = await self.store.get(request_hash)
            if recorded is None:
                raise LookupError("recorded LLM response unavailable")
            return recorded.structured_response
        if mode == ReplayLLMMode.DETERMINISTIC_BASELINE:
            if self.baseline is None:
                raise LookupError("deterministic replay baseline unavailable")
            return self.baseline
        raise ValueError("fresh-model replay must use the explicitly configured existing LLM provider")


class PointInTimeReplay:
    def validate_state(self, state: UnifiedMarketState, replay_cursor: datetime) -> UnifiedMarketState:
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        if state.knowledge_cutoff > replay_cursor:
            raise ValueError("replay state contains future knowledge")
        if any(item.available_at > replay_cursor for item in state.evidence):
            raise ValueError("replay evidence was unavailable at the replay cursor")
        return state

    @staticmethod
    def manifest(
        states: tuple[UnifiedMarketState, ...],
        *,
        quantitative_model_version: str,
        prompt_version: str,
        policy_versions: dict[str, str],
        setup_family_registry_version: str,
        spread_assumption: float,
        slippage_assumption: float,
        llm_mode: ReplayLLMMode,
    ) -> dict[str, Any]:
        ordered = tuple(sorted(states, key=lambda item: item.market_data_boundary))
        return {
            "state_hashes": [item.state_hash for item in ordered],
            "quantitative_model_version": quantitative_model_version,
            "prompt_version": prompt_version,
            "policy_versions": policy_versions,
            "setup_family_registry_version": setup_family_registry_version,
            "spread_assumption": spread_assumption,
            "slippage_assumption": slippage_assumption,
            "llm_mode": llm_mode.value,
            "deterministic": llm_mode != ReplayLLMMode.FRESH_MODEL,
        }
