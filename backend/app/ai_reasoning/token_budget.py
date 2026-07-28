"""Central, model-aware token budgets for TEN's external AI boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from math import ceil
from typing import Any


class OutputProfile(StrEnum):
    COMPACT = "compact"
    STANDARD = "standard"
    EXPANDED = "expanded"
    COMPACT_RETRY = "compact_retry"


@dataclass(frozen=True)
class OutputLimits:
    target_tokens: int
    hard_limit: int


@dataclass(frozen=True)
class TokenEstimate:
    tokens: int
    method: str = "conservative_utf8_bytes_divided_by_3"
    exact: bool = False


@dataclass(frozen=True)
class TokenBudgetPlan:
    model: str
    model_context_limit: int
    reserved_system_tokens: int
    reserved_output_tokens: int
    maximum_input_tokens: int
    target_output_tokens: int
    hard_output_limit: int
    schema_token_cost: int
    context_token_cost: int
    prompt_token_cost: int
    safety_margin_tokens: int
    output_profile: OutputProfile
    estimator: str
    context_sections_included: tuple[str, ...]
    context_sections_omitted: tuple[str, ...]

    @property
    def estimated_input_tokens(self) -> int:
        return self.schema_token_cost + self.context_token_cost + self.prompt_token_cost

    @property
    def input_budget_utilization_percent(self) -> float:
        return round(
            self.estimated_input_tokens / max(1, self.maximum_input_tokens) * 100,
            2,
        )


class TokenBudgetManager:
    """The only source of output limits and preflight token estimates."""

    _PROFILE_LIMITS = {
        OutputProfile.COMPACT: OutputLimits(target_tokens=900, hard_limit=1400),
        OutputProfile.STANDARD: OutputLimits(target_tokens=1500, hard_limit=2100),
        OutputProfile.EXPANDED: OutputLimits(target_tokens=2400, hard_limit=3200),
        OutputProfile.COMPACT_RETRY: OutputLimits(target_tokens=650, hard_limit=1000),
    }

    def __init__(
        self,
        *,
        model: str,
        output_profile: OutputProfile | str = OutputProfile.COMPACT,
        model_context_limit: int = 8192,
        maximum_input_tokens: int = 3500,
        target_output_tokens: int | None = None,
        hard_output_limit: int | None = None,
        safety_margin_tokens: int = 256,
    ) -> None:
        self.model = model
        self.output_profile = OutputProfile(output_profile)
        self.model_context_limit = model_context_limit
        self.maximum_input_tokens = maximum_input_tokens
        self.target_output_tokens = target_output_tokens
        self.hard_output_limit = hard_output_limit
        self.safety_margin_tokens = safety_margin_tokens

    @staticmethod
    def estimate(value: str | bytes | dict[str, Any] | list[Any]) -> TokenEstimate:
        if isinstance(value, bytes):
            encoded = value
        elif isinstance(value, str):
            encoded = value.encode()
        else:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode()
        return TokenEstimate(tokens=ceil(len(encoded) / 3))

    def limits(self, profile: OutputProfile | str | None = None) -> OutputLimits:
        selected = OutputProfile(profile or self.output_profile)
        defaults = self._PROFILE_LIMITS[selected]
        if selected != self.output_profile:
            return defaults
        return OutputLimits(
            target_tokens=self.target_output_tokens or defaults.target_tokens,
            hard_limit=self.hard_output_limit or defaults.hard_limit,
        )

    def plan(
        self,
        *,
        system_prompt: str,
        context: dict[str, Any],
        schema: dict[str, Any],
        profile: OutputProfile | str | None = None,
        included_sections: tuple[str, ...] = (),
        omitted_sections: tuple[str, ...] = (),
    ) -> TokenBudgetPlan:
        selected = OutputProfile(profile or self.output_profile)
        limits = self.limits(selected)
        prompt_cost = self.estimate(system_prompt)
        context_cost = self.estimate(context)
        schema_cost = self.estimate(schema)
        reserved_system = prompt_cost.tokens
        available_from_model = (
            self.model_context_limit
            - limits.hard_limit
            - self.safety_margin_tokens
            - reserved_system
        )
        maximum_input = min(self.maximum_input_tokens, max(0, available_from_model))
        return TokenBudgetPlan(
            model=self.model,
            model_context_limit=self.model_context_limit,
            reserved_system_tokens=reserved_system,
            reserved_output_tokens=limits.hard_limit,
            maximum_input_tokens=maximum_input,
            target_output_tokens=limits.target_tokens,
            hard_output_limit=limits.hard_limit,
            schema_token_cost=schema_cost.tokens,
            context_token_cost=context_cost.tokens,
            prompt_token_cost=prompt_cost.tokens,
            safety_margin_tokens=self.safety_margin_tokens,
            output_profile=selected,
            estimator=prompt_cost.method,
            context_sections_included=included_sections,
            context_sections_omitted=omitted_sections,
        )
