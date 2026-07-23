"""Versioned registry of genuine technical, execution, and safety gates."""

from __future__ import annotations

from .models import HardGateDefinition


def _gate(
    gate_id: str,
    category: str,
    required_inputs: tuple[str, ...],
    *,
    severity: str = "blocking",
    behavior: str = "reject",
    actions: tuple[str, ...] = ("new_signal", "monitoring_update"),
    setups: tuple[str, ...] = ("*",),
) -> HardGateDefinition:
    return HardGateDefinition(
        gate_id=gate_id,
        gate_version="1.0",
        category=category,
        applicable_actions=actions,
        applicable_setup_families=setups,
        required_inputs=required_inputs,
        evaluator=f"evaluate_{gate_id}",
        severity=severity,
        block_behavior=behavior,
        reason_codes=(gate_id,),
        configuration_source="configs/ai_guardrails.yaml",
    )


class HardGateRegistry:
    """Static evaluator identities with versioned, configurable policy inputs."""

    version = "hard_gates_v1"

    def __init__(self) -> None:
        gates = (
            _gate("market_state_consistent", "technical", ("market_state",)),
            _gate("authoritative_data_fresh", "technical", ("market_state",), behavior="postpone"),
            _gate("point_in_time_valid", "technical", ("market_state", "quantitative_forecast", "ai_forecast")),
            _gate("future_data_absent", "technical", ("market_state", "quantitative_forecast")),
            _gate("market_open", "technical", ("execution_context.market_open",), behavior="postpone"),
            _gate("quantitative_forecast_available", "technical", ("quantitative_forecast.status",)),
            _gate("ai_forecast_valid", "technical", ("ai_forecast.status", "ai_forecast.validation_passed")),
            _gate("ai_proposal_valid", "technical", ("ai_proposal",)),
            _gate("mandatory_setup_evidence", "technical", ("setup_family", "market_state.evidence")),
            _gate("signal_lifecycle_valid", "technical", ("managed_signal.state",)),
            _gate("duplicate_structural_opportunity", "technical", ("structural_opportunity_key",)),
            _gate("publication_service_available", "technical", ("execution_context.publication_service_available",), behavior="postpone"),
            _gate("persistence_state_valid", "technical", ("execution_context.persistence_available",), behavior="postpone"),
            _gate("entry_geometry_valid", "execution", ("ai_proposal.entry_zone",)),
            _gate("stop_geometry_valid", "execution", ("ai_proposal.stop_loss",)),
            _gate("target_geometry_valid", "execution", ("ai_proposal.take_profit_levels",)),
            _gate("signal_type_supported", "execution", ("ai_proposal.entry_type",)),
            _gate("spread_within_session_limit", "execution", ("execution_context.spread", "execution_context.session"), behavior="postpone"),
            _gate("price_precision_valid", "execution", ("price_levels",)),
            _gate("entry_distance_valid", "execution", ("execution_context.current_price", "ai_proposal.entry_zone"), behavior="postpone"),
            _gate("setup_not_expired", "execution", ("ai_proposal.expires_at",), behavior="postpone"),
            _gate("execution_context_available", "execution", ("execution_context",)),
            _gate("absolute_risk_to_reward", "risk", ("price_geometry",)),
            _gate("maximum_risk_per_signal", "risk", ("account_risk",)),
            _gate("maximum_simultaneous_exposure", "risk", ("account_risk",)),
            _gate("maximum_daily_loss", "risk", ("account_risk",)),
            _gate("maximum_aggregate_drawdown", "risk", ("account_risk",)),
            _gate("conflicting_active_exposure", "risk", ("account_risk",)),
            _gate("economic_event_blackout", "risk", ("economic_context",), behavior="postpone"),
            _gate("stop_distance_safe", "risk", ("price_geometry",)),
            _gate("position_size_valid", "risk", ("account_risk",)),
        )
        self._gates = {item.gate_id: item for item in gates}

    def all(self) -> tuple[HardGateDefinition, ...]:
        return tuple(self._gates.values())

    def get(self, gate_id: str) -> HardGateDefinition:
        return self._gates[gate_id]
