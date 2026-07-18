from backend.app.core.exceptions import TenError


class AIScoringError(TenError):
    """Base class for sanitized AI Scoring failures."""


class AIScoringConfigurationError(AIScoringError):
    pass


class AIScoringInputError(AIScoringError):
    pass


class AIScoringPersistenceError(AIScoringError):
    pass


class AIScoringPointInTimeError(AIScoringError):
    pass


class AIScoringDependencyError(AIScoringError):
    pass
