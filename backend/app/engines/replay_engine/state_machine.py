from __future__ import annotations

from .exceptions import ReplayTransitionError
from .models import ReplayStatus

TRANSITIONS: dict[ReplayStatus, frozenset[ReplayStatus]] = {
    ReplayStatus.CREATED: frozenset({ReplayStatus.VALIDATING}),
    ReplayStatus.VALIDATING: frozenset({ReplayStatus.READY, ReplayStatus.FAILED}),
    ReplayStatus.READY: frozenset({ReplayStatus.RUNNING, ReplayStatus.CANCELLING}),
    ReplayStatus.RUNNING: frozenset({ReplayStatus.PAUSING, ReplayStatus.CANCELLING, ReplayStatus.COMPLETED, ReplayStatus.FAILED}),
    ReplayStatus.PAUSING: frozenset({ReplayStatus.PAUSED, ReplayStatus.FAILED}),
    ReplayStatus.PAUSED: frozenset({ReplayStatus.RESUMING, ReplayStatus.CANCELLING}),
    ReplayStatus.RESUMING: frozenset({ReplayStatus.RUNNING, ReplayStatus.FAILED}),
    ReplayStatus.CANCELLING: frozenset({ReplayStatus.CANCELLED, ReplayStatus.FAILED}),
    ReplayStatus.FAILED: frozenset({ReplayStatus.RECOVERING}),
    ReplayStatus.RECOVERING: frozenset({ReplayStatus.READY, ReplayStatus.FAILED}),
    ReplayStatus.CANCELLED: frozenset(),
    ReplayStatus.COMPLETED: frozenset(),
}


def validate_transition(current: ReplayStatus, target: ReplayStatus) -> None:
    if target not in TRANSITIONS[current]:
        raise ReplayTransitionError(f"invalid replay transition: {current.value} -> {target.value}")
