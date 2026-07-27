"""Deterministic authorization for monitored signal changes."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from backend.app.ai_reasoning.models import Direction, ManagedSignal, ManagedSignalState, ProposalAction


class LevelRevisionService(Protocol):
    async def revise_level(
        self,
        signal: ManagedSignal,
        *,
        level_type: str,
        new_value: float,
        reason: str,
        evidence_ids: tuple[UUID, ...],
        approved_rule: str,
    ) -> tuple[ManagedSignal, Any]: ...


class MonitoringAdjustmentPolicy:
    """No mutation is legal unless both the flag and an explicit policy rule allow it."""

    _ALLOWED_ACTION_STATES = {
        ProposalAction.ADJUST_ENTRY: {
            ManagedSignalState.PROPOSED,
            ManagedSignalState.CONFIRMED,
            ManagedSignalState.WAITING_FOR_ENTRY,
        },
        ProposalAction.CANCEL_SETUP: {
            ManagedSignalState.PROPOSED,
            ManagedSignalState.CONFIRMED,
            ManagedSignalState.WAITING_FOR_ENTRY,
            ManagedSignalState.TEMPORARILY_BLOCKED,
        },
        ProposalAction.INVALIDATE_SIGNAL: {
            ManagedSignalState.PROPOSED,
            ManagedSignalState.CONFIRMED,
            ManagedSignalState.WAITING_FOR_ENTRY,
            ManagedSignalState.ACTIVE,
            ManagedSignalState.PARTIALLY_REALIZED,
        },
        ProposalAction.REDUCE_RISK: {
            ManagedSignalState.WAITING_FOR_ENTRY,
            ManagedSignalState.ACTIVE,
            ManagedSignalState.PARTIALLY_REALIZED,
        },
        ProposalAction.TAKE_PARTIAL_PROFIT: {
            ManagedSignalState.ACTIVE,
            ManagedSignalState.PARTIALLY_REALIZED,
            ManagedSignalState.TP1_HIT,
        },
        ProposalAction.CLOSE_SIGNAL: {
            ManagedSignalState.ACTIVE,
            ManagedSignalState.PARTIALLY_REALIZED,
            ManagedSignalState.TP1_HIT,
            ManagedSignalState.TP2_HIT,
        },
    }

    def __init__(self, *, enabled: bool, policy_version: str) -> None:
        self.enabled = enabled
        self.policy_version = policy_version

    def authorize(self, action: ProposalAction, signal: ManagedSignal) -> str:
        if not self.enabled:
            raise PermissionError("ai_signal_adjustments is disabled")
        allowed_states = self._ALLOWED_ACTION_STATES.get(action)
        if allowed_states is None or signal.state not in allowed_states:
            raise ValueError(f"monitoring action {action.value} is not legal from {signal.state.value}")
        return f"{self.policy_version}:{action.value}"

    async def revise_stop(
        self,
        lifecycle: LevelRevisionService,
        signal: ManagedSignal,
        *,
        new_stop: float,
        reason: str,
        evidence_ids: tuple[UUID, ...],
    ) -> tuple[ManagedSignal, Any]:
        self.authorize(ProposalAction.REDUCE_RISK, signal)
        if signal.stop_loss is None:
            raise ValueError("signal has no existing Stop Loss")
        protective = (
            signal.direction == Direction.BUY and new_stop >= signal.stop_loss
        ) or (
            signal.direction == Direction.SELL and new_stop <= signal.stop_loss
        )
        if not protective:
            raise ValueError("unauthorized Stop widening")
        return await lifecycle.revise_level(
            signal,
            level_type="stop_loss",
            new_value=new_stop,
            reason=reason,
            evidence_ids=evidence_ids,
            approved_rule=f"{self.policy_version}:protective_stop_only",
        )
