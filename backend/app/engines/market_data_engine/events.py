"""Market-data domain events published through TEN's existing event bus."""

from backend.app.events.models import Event


class MarketOpened(Event):
    pass


class MarketClosed(Event):
    pass


class SessionChanged(Event):
    pass


class ProviderChanged(Event):
    pass


class ProviderRecovered(Event):
    pass


class GapDetected(Event):
    pass


class HistoricalUpdated(Event):
    pass


class RealtimeUpdated(Event):
    pass


class NewCandle(Event):
    pass


class DataCorrupted(Event):
    pass


class QualityChanged(Event):
    pass
