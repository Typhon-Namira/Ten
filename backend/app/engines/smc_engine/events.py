"""Typed Milestone 2A events carried by TEN's existing Event Bus."""

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
