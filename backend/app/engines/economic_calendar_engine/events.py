from backend.app.events.models import Event


class EconomicCalendarSyncStarted(Event):
    pass


class EconomicCalendarSyncCompleted(Event):
    pass


class EconomicCalendarSyncFailed(Event):
    pass


class EconomicEventDiscovered(Event):
    pass


class EconomicEventUpdated(Event):
    pass


class EconomicEventReleased(Event):
    pass


class EconomicEventRevised(Event):
    pass


class EconomicEventCorrected(Event):
    pass


class EconomicEventCancelled(Event):
    pass


class EconomicEventPostponed(Event):
    pass


class EconomicEventRescheduled(Event):
    pass


class EconomicEventConflictDetected(Event):
    pass


class EconomicEventConflictResolved(Event):
    pass


class EconomicEventWindowEntered(Event):
    pass


class EconomicEventImminent(Event):
    pass


class EconomicEventWindowExited(Event):
    pass


class EconomicEventClusterDetected(Event):
    pass


class EconomicCalendarSnapshotCreated(Event):
    pass


class EconomicCalendarReplayCompleted(Event):
    pass


class EconomicCalendarRecoveryCompleted(Event):
    pass


class EconomicCalendarDegraded(Event):
    pass


class EconomicCalendarProviderRecovered(Event):
    pass


class EconomicCalendarProviderStale(Event):
    pass


class EconomicCalendarProviderRequestCompleted(Event):
    """One structured log line per provider per sync attempt — provider name, connection state,
    HTTP status, latency, retry count, and outcome — so live logs show exactly what each provider
    request did instead of only a pass/fail sync-level summary."""
