from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from backend.app.events import Event

from .models import ReplayRequest, ReplayState


class ReplayEngine(ABC):
    """Historical replay/simulation contract only; execution is not implemented."""

    name = "replay"
    version = "1.0.0"
    compatibility_version = "1.0"

    @abstractmethod
    async def create(self, request: ReplayRequest) -> ReplayState:
        """Create a future deterministic replay session."""

    @abstractmethod
    def events(self, state: ReplayState) -> AsyncIterator[Event]:
        """Yield historical events using a future replay clock."""
