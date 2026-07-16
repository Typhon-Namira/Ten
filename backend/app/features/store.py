"""Feature store abstraction and deterministic in-memory adapter."""

import asyncio
from abc import ABC, abstractmethod
from uuid import UUID

from .models import FeatureRecord, FeatureSnapshot


class FeatureStore(ABC):
    @abstractmethod
    async def write(self, feature: FeatureRecord) -> None:
        """Persist one immutable versioned feature record."""

    @abstractmethod
    async def snapshot(self, correlation_id: UUID) -> FeatureSnapshot:
        """Return the latest feature value per namespace for an analysis run."""


class InMemoryFeatureStore(FeatureStore):
    def __init__(self) -> None:
        self._records: list[FeatureRecord] = []
        self._lock = asyncio.Lock()

    async def write(self, feature: FeatureRecord) -> None:
        async with self._lock:
            self._records.append(feature)

    async def snapshot(self, correlation_id: UUID) -> FeatureSnapshot:
        async with self._lock:
            records = [record for record in self._records if record.correlation_id == correlation_id]
        features: dict[str, dict[str, object]] = {}
        versions: dict[str, str] = {}
        for record in records:
            features[record.namespace] = record.values
            versions[record.engine_name] = record.engine_version
        return FeatureSnapshot(correlation_id=correlation_id, features=features, engine_versions=versions)
