from abc import ABC, abstractmethod

from .models import MemoryContext, OutcomeRecord, ReasoningRecord, SignalMemoryRecord


class SignalMemory(ABC):
    @abstractmethod
    async def remember_signal(self, record: SignalMemoryRecord) -> None:
        """Persist a historical signal context."""


class OutcomeMemory(ABC):
    @abstractmethod
    async def remember_outcome(self, record: OutcomeRecord) -> None:
        """Persist a future observed outcome."""


class ContextRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, limit: int = 10) -> MemoryContext:
        """Retrieve relevant historical context."""


class ReasoningHistory(ABC):
    @abstractmethod
    async def remember_reasoning(self, record: ReasoningRecord) -> None:
        """Persist model reasoning provenance."""


class AIMemory(SignalMemory, OutcomeMemory, ContextRetriever, ReasoningHistory, ABC):
    """Composite AI memory port; no adapter is provided in this foundation."""
