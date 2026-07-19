from backend.app.events import Event


class ReplayLifecycleEvent(Event):
    """Bounded operational event; historical analytical events use ReplayEventBus."""


class ReplayCreated(ReplayLifecycleEvent):
    pass


class ReplayStarted(ReplayLifecycleEvent):
    pass


class ReplayPaused(ReplayLifecycleEvent):
    pass


class ReplayResumed(ReplayLifecycleEvent):
    pass


class ReplayCheckpointed(ReplayLifecycleEvent):
    pass


class ReplayCompleted(ReplayLifecycleEvent):
    pass


class ReplayFailed(ReplayLifecycleEvent):
    pass


class ReplayCancelled(ReplayLifecycleEvent):
    pass
