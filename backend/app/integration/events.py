from backend.app.events import Event


class CanonicalIntegrationEvent(Event):
    """Outbox-delivered canonical integration envelope."""


class IntegrationSnapshotReady(Event):
    """A coherent point-in-time evidence barrier is ready."""


class OperationalSignalGenerated(Event):
    """A persisted analytical Signal Decision is dashboard-visible."""
