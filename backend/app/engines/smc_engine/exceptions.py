"""Structured SMC failure categories."""

from backend.app.core.exceptions import EngineError


class SMCError(EngineError):
    pass


class InvalidSMCInput(SMCError):
    pass


class InsufficientHistory(SMCError):
    pass


class SMCStateConflict(SMCError):
    pass


class SMCReplayMismatch(SMCError):
    pass


class SMCProcessingError(SMCError):
    pass


class SMCPersistenceError(SMCError):
    pass
