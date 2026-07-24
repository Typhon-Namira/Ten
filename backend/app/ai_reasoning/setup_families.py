"""Versioned registry of setup-specific evidence and lifecycle policies."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import YamlConfigRepository


class SetupFamilyDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    setup_family_id: str
    version: str
    applicable_regimes: tuple[str, ...]
    unsuitable_regimes: tuple[str, ...] = ()
    mandatory_evidence: tuple[str, ...]
    supporting_evidence: tuple[str, ...] = ()
    contradictory_evidence: tuple[str, ...] = ()
    optional_evidence: tuple[str, ...] = ()
    entry_model: str
    invalidation_model: str
    stop_loss_model: str
    target_model: str
    expected_horizon: str
    expiry_policy: str
    monitoring_policy: str
    minimum_evidence_completeness: float = Field(ge=0, le=1)
    permitted_signal_actions: tuple[str, ...]


class SetupFamilyRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    families: tuple[SetupFamilyDefinition, ...]


class SetupFamilyRegistry:
    _ALIASES = {
        "trend_following": "trend_continuation",
        "pullback": "pullback_continuation",
        "liquidity_sweep": "liquidity_sweep_reversal",
        "breakout": "breakout_retest",
        "order_block": "order_block_retest",
        "fvg": "fvg_continuation",
        "fair_value_gap": "fvg_continuation",
        "displacement": "displacement_continuation",
        "range_reversal": "range_boundary_reversal",
    }

    def __init__(self, config: SetupFamilyRegistryConfig) -> None:
        self.version = config.version
        self._families = {item.setup_family_id: item for item in config.families}
        if len(self._families) != len(config.families):
            raise ValueError("setup-family IDs must be unique")

    @classmethod
    def from_yaml(cls, configs: YamlConfigRepository) -> SetupFamilyRegistry:
        return cls(configs.load_model("ai_setup_families", SetupFamilyRegistryConfig))

    def get(self, setup_family_id: str) -> SetupFamilyDefinition | None:
        return self._families.get(setup_family_id)

    def all(self) -> tuple[SetupFamilyDefinition, ...]:
        return tuple(self._families.values())

    def canonical_id(self, value: object) -> tuple[str | None, bool]:
        """Resolve only exact IDs or explicit, unambiguous provider aliases."""

        if not isinstance(value, str) or not value.strip():
            return None, False
        normalized = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")
        if normalized in self._families:
            return normalized, normalized != value
        canonical = self._ALIASES.get(normalized)
        if canonical in self._families:
            return canonical, True
        return None, False

    def validate_requirements(
        self,
        setup_family_id: str,
        available_evidence_kinds: set[str],
        evidence_completeness: float,
        action: str,
    ) -> tuple[str, ...]:
        family = self.get(setup_family_id)
        if family is None:
            return ("unknown_setup_family",)
        errors = [
            f"missing_setup_evidence:{kind}"
            for kind in family.mandatory_evidence
            if kind not in available_evidence_kinds
        ]
        if evidence_completeness < family.minimum_evidence_completeness:
            errors.append("setup_evidence_completeness_below_minimum")
        if action not in family.permitted_signal_actions:
            errors.append("unsupported_action_for_setup_family")
        return tuple(errors)
