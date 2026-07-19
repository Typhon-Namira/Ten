from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .models import HistoricalEvent, ReplayRequest, ReplayState
from .ordering import merge_event_sources
from .service import ReplayService
from .sources import HistoricalEventQuery


class ReplayEngine(ABC):
    """Historical reconstruction contract. It is explicitly not a backtester."""

    name = "replay"
    version = "1.0.0"
    compatibility_version = "1.0"

    @abstractmethod
    async def create(self, request: ReplayRequest) -> ReplayState: ...

    @abstractmethod
    def events(self, state: ReplayState) -> AsyncIterator[HistoricalEvent]: ...


class ProductionReplayEngine(ReplayEngine):
    def __init__(self, service: ReplayService) -> None:
        self.service = service

    async def create(self, request: ReplayRequest) -> ReplayState:
        return await self.service.create(request)

    async def events(self, state: ReplayState) -> AsyncIterator[HistoricalEvent]:
        sources = self.service.coordinator.sources.resolve(state.request.source_filters.source_names)
        query = HistoricalEventQuery(state.replay_id, state.request, state.last_ordering_key, self.service.config.processing.source_batch_size)
        streams = [source.stream(query) for source in sources]
        async for event in merge_event_sources(streams):
            yield event
