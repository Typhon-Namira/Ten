from __future__ import annotations

from typing import Protocol

from .exceptions import SignalDecisionConfigurationError
from .models import SignalDecision, SignalDecisionInput


class DecisionPolicy(Protocol):
    name: str
    version: str

    def evaluate(self, decision_input: SignalDecisionInput) -> SignalDecision: ...


class DecisionPolicyRegistry:
    def __init__(self) -> None:
        self._policies: dict[tuple[str, str], DecisionPolicy] = {}

    def register(self, policy: DecisionPolicy) -> None:
        key = (policy.name, policy.version)
        if key in self._policies:
            raise SignalDecisionConfigurationError(f"duplicate decision policy: {policy.name}:{policy.version}")
        self._policies[key] = policy

    def get(self, name: str, version: str) -> DecisionPolicy:
        try:
            return self._policies[(name, version)]
        except KeyError as exc:
            raise SignalDecisionConfigurationError(f"unknown decision policy: {name}:{version}") from exc
