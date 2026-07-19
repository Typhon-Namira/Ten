from backend.app.core.exceptions import TenError


class ReplayError(TenError):
    """Base sanitized Replay Engine error."""


class ReplayConfigurationError(ReplayError):
    pass


class ReplayValidationError(ReplayError):
    pass


class ReplayNotFound(ReplayError):
    pass


class ReplayTransitionError(ReplayError):
    pass


class ReplayConcurrencyError(ReplayError):
    pass


class ReplayPointInTimeError(ReplayError):
    pass


class ReplayOrderingError(ReplayError):
    pass


class ReplayIsolationError(ReplayError):
    pass


class ReplayCheckpointError(ReplayError):
    pass


class ReplayPersistenceError(ReplayError):
    pass
