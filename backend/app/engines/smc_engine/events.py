"""Typed SMC Production 1.0 events carried by TEN's existing Event Bus."""

from backend.app.events.models import Event


class SwingCandidateDetected(Event):
    pass


class SwingConfirmed(Event):
    pass


class SwingInvalidated(Event):
    pass


class StructureDirectionChanged(Event):
    pass


class BOSDetected(Event):
    pass


class CHOCHDetected(Event):
    pass


class MSSDetected(Event):
    pass


class StructureInvalidated(Event):
    pass


class SMCAnalysisUpdated(Event):
    pass


class SMCReplayCompleted(Event):
    pass


class SMCInputDegraded(Event):
    pass


class DisplacementDetected(Event):
    pass


class ImbalanceDetected(Event):
    pass


class OrderBlockDetected(Event):
    pass


class BreakerBlockDetected(Event):
    pass


class MitigationBlockDetected(Event):
    pass


class LiquidityVoidDetected(Event):
    pass


class DealingRangeUpdated(Event):
    pass


class SMCObjectLifecycleChanged(Event):
    pass


class MultiTimeframeContextUpdated(Event):
    pass
