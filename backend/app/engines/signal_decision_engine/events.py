from backend.app.events import Event


class SignalDecisionEligible(Event):
    pass


class SignalDecisionObserveOnly(Event):
    pass


class SignalDecisionBlocked(Event):
    pass


class SignalDecisionInsufficientEvidence(Event):
    pass


class SignalDecisionInvalid(Event):
    pass


class SignalDecisionExpired(Event):
    pass


class SignalDecisionSuperseded(Event):
    pass


class SignalDecisionFailed(Event):
    pass
