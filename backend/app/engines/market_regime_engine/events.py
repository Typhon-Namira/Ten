from backend.app.events.models import Event


class MarketRegimeSnapshotCreated(Event):
    pass


class MarketRegimeChanged(Event):
    pass


class TrendRegimeChanged(Event):
    pass


class VolatilityRegimeChanged(Event):
    pass


class AuctionRegimeChanged(Event):
    pass


class CompressionDetected(Event):
    pass


class ExpansionDetected(Event):
    pass


class RegimeTransitionStarted(Event):
    pass


class RegimeTransitionConfirmed(Event):
    pass


class RegimeTransitionFailed(Event):
    pass


class RegimeTransitionInvalidated(Event):
    pass


class RegimeWeakeningDetected(Event):
    pass


class RegimeReversalRiskDetected(Event):
    pass


class MultiTimeframeConflictDetected(Event):
    pass


class CrossSessionHandoffDetected(Event):
    pass


class MarketRegimeReplayCompleted(Event):
    pass


class MarketRegimeRecoveryCompleted(Event):
    pass


class MarketRegimeDegraded(Event):
    pass


class MarketRegimeDependencyRecovered(Event):
    pass
