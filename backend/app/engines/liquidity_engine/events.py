from backend.app.events.models import Event


class EqualHighClusterConfirmed(Event):
    pass


class EqualLowClusterConfirmed(Event):
    pass


class LiquidityPoolCreated(Event):
    pass


class LiquidityPoolUpdated(Event):
    pass


class LiquidityPoolApproached(Event):
    pass


class LiquidityPoolTouched(Event):
    pass


class LiquidityPoolPartiallySwept(Event):
    pass


class LiquidityPoolSwept(Event):
    pass


class LiquidityGrabDetected(Event):
    pass


class LiquidityRaidDetected(Event):
    pass


class StopHuntClassified(Event):
    pass


class FalseBreakConfirmed(Event):
    pass


class LiquidityReclaimed(Event):
    pass


class LiquidityPoolConsumed(Event):
    pass


class LiquidityPoolInvalidated(Event):
    pass


class LiquidityPoolExpired(Event):
    pass


class SessionLiquidityUpdated(Event):
    pass


class ReferenceLiquidityCreated(Event):
    pass


class LiquidityConfluenceUpdated(Event):
    pass


class LiquidityTargetRankingUpdated(Event):
    pass


class LiquidityInputDegraded(Event):
    pass


class LiquidityCheckpointRecovered(Event):
    pass


class LiquidityReplayCompleted(Event):
    pass


class LiquidityAnalysisUpdated(Event):
    pass
