from backend.app.events.models import Event


class VolumeProfileInitialized(Event):
    pass


class VolumeProfileDeveloping(Event):
    pass


class VolumeProfileUpdated(Event):
    pass


class VolumeProfileCompleted(Event):
    pass


class VolumeProfilePublished(Event):
    pass


class VolumeProfileDegraded(Event):
    pass


class VolumeProfileFailed(Event):
    pass


class PointOfControlChanged(Event):
    pass


class PointOfControlMigrated(Event):
    pass


class ValueAreaChanged(Event):
    pass


class ValueAreaExpanded(Event):
    pass


class ValueAreaContracted(Event):
    pass


class HighVolumeNodeDetected(Event):
    pass


class LowVolumeNodeDetected(Event):
    pass


class VolumeShelfDetected(Event):
    pass


class VolumeGapDetected(Event):
    pass


class ProfileShapeChanged(Event):
    pass


class AnchoredProfileCreated(Event):
    pass


class CompositeProfileCompleted(Event):
    pass


class ProfileReferenceTested(Event):
    pass


class ProfileInvalidated(Event):
    pass


class VolumeProfileCheckpointRecovered(Event):
    pass


class VolumeProfileReplayCompleted(Event):
    pass


class VolumeProfileAnalysisUpdated(Event):
    pass
