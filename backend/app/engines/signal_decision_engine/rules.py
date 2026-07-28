from __future__ import annotations

from dataclasses import dataclass

from .exceptions import SignalDecisionConfigurationError
from .models import RuleCategory


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    category: RuleCategory
    version: str = "1.0.0"


class RuleRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, RuleDefinition] = {}

    def register(self, definition: RuleDefinition) -> None:
        if definition.rule_id in self._rules:
            raise SignalDecisionConfigurationError(f"duplicate decision rule: {definition.rule_id}")
        self._rules[definition.rule_id] = definition

    def get(self, rule_id: str) -> RuleDefinition:
        try:
            return self._rules[rule_id]
        except KeyError as exc:
            raise SignalDecisionConfigurationError(f"unknown decision rule: {rule_id}") from exc

    def definitions(self) -> tuple[RuleDefinition, ...]:
        return tuple(self._rules[key] for key in sorted(self._rules))


def production_rule_registry() -> RuleRegistry:
    registry = RuleRegistry()
    definitions = (
        ("policy.integrity", RuleCategory.POLICY_INTEGRITY),
        ("source.snapshot_integrity", RuleCategory.SOURCE_INTEGRITY),
        ("source.point_in_time", RuleCategory.SOURCE_INTEGRITY),
        ("freshness.ai_score", RuleCategory.FRESHNESS),
        ("strength.minimum", RuleCategory.DIRECTIONAL_STRENGTH),
        ("confidence.minimum", RuleCategory.CONFIDENCE),
        ("risk.maximum", RuleCategory.RISK),
        ("quality.minimum", RuleCategory.DATA_QUALITY),
        ("alignment.minimum", RuleCategory.ALIGNMENT),
        ("ai.signal_authority", RuleCategory.ALIGNMENT),
        ("conflict.maximum", RuleCategory.CONFLICT),
        ("economic_event.window", RuleCategory.ECONOMIC_EVENT),
        ("regime.allowed", RuleCategory.MARKET_REGIME),
        ("dependency.health", RuleCategory.DEPENDENCY_HEALTH),
        ("duplicate.active_equivalent", RuleCategory.DUPLICATE),
        ("cooldown.instrument_direction", RuleCategory.COOLDOWN),
        ("reversal.opposite_direction_lock", RuleCategory.REVERSAL),
        ("temporal.validity", RuleCategory.TEMPORAL_VALIDITY),
    )
    for rule_id, category in definitions:
        registry.register(RuleDefinition(rule_id, category))
    return registry
