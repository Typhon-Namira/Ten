from backend.app.core.exceptions import TenError


class SignalDecisionError(TenError):
    pass


class SignalDecisionConfigurationError(SignalDecisionError):
    pass


class SignalDecisionInputError(SignalDecisionError):
    pass


class SignalDecisionSnapshotNotFound(SignalDecisionError):
    pass


class SignalDecisionPersistenceError(SignalDecisionError):
    pass
