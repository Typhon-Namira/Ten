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

    async def history(self, *, mode: str, instrument: str, timeframe: str, limit: int = 100) -> tuple[FeatureRecord, ...]:
        """Return bounded immutable feature history for one analytical key."""
        raise NotImplementedError


class InMemoryFeatureStore(FeatureStore):
    def __init__(self) -> None:
        self._records: list[FeatureRecord] = []
        self._identifiers: set[UUID] = set()
        self._lock = asyncio.Lock()

    async def write(self, feature: FeatureRecord) -> None:
        async with self._lock:
            if feature.feature_id in self._identifiers:
                return
            self._records.append(feature)
            self._identifiers.add(feature.feature_id)

    async def snapshot(self, correlation_id: UUID) -> FeatureSnapshot:
        async with self._lock:
            records = [record for record in self._records if record.correlation_id == correlation_id]
        features: dict[str, dict[str, object]] = {}
        versions: dict[str, str] = {}
        for record in records:
            features[record.namespace] = record.values
            versions[record.engine_name] = record.engine_version
        return FeatureSnapshot(correlation_id=correlation_id, features=features, engine_versions=versions)

    async def history(self, *, mode: str, instrument: str, timeframe: str, limit: int = 100) -> tuple[FeatureRecord, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("feature history limit must be between 1 and 1000")
        async with self._lock:
            values = [item for item in self._records if item.mode == mode and item.instrument == instrument and item.timeframe == timeframe]
        values.sort(key=lambda item: (item.effective_at or item.created_at, str(item.feature_id)), reverse=True)
        return tuple(values[:limit])
