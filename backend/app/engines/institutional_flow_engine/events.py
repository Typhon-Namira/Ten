from backend.app.events.models import Event


class InstitutionalFlowInitialized(Event):
    pass


class InstitutionalFlowUpdated(Event):
    pass


class InstitutionalFlowDegraded(Event):
    pass


class ParticipationChanged(Event):
    pass


class InitiativeActivityInferred(Event):
    pass


class ResponsiveActivityInferred(Event):
    pass


class AbsorptionLikeBehaviorInferred(Event):
    pass


class ExhaustionLikeBehaviorInferred(Event):
    pass


class InventoryBehaviorInferred(Event):
    pass


class CampaignPhaseChanged(Event):
    pass


class DirectionalPressureChanged(Event):
    pass


class CrossSessionFlowAnalyzed(Event):
    pass


class InstitutionalFlowCheckpointRecovered(Event):
    pass


class InstitutionalFlowReplayCompleted(Event):
    pass
