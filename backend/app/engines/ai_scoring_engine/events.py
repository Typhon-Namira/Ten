from backend.app.events import Event


class AIScoringCompleted(Event):
    pass


class AIScoringDegraded(Event):
    pass


class AIScoringInsufficientEvidence(Event):
    pass


class AIScoringConflictDetected(Event):
    pass
