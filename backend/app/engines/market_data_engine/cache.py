"""Bounded memory and durable file caches with observable hit statistics."""

import asyncio
import json
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from .models import Candle


class CacheStatistics(BaseModel):
    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    invalidations: int = 0
    last_write_at: datetime | None = None

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class MarketDataCache:
    def __init__(self, directory: Path, *, max_entries: int = 10_000) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.statistics = CacheStatistics()
        self._memory: OrderedDict[str, tuple[datetime, list[Candle]]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> list[Candle] | None:
        async with self._lock:
            entry = self._memory.get(key)
            if entry and entry[0] > datetime.now(UTC):
                self._memory.move_to_end(key)
                self.statistics.hits += 1
                return list(entry[1])
            if entry:
                self._memory.pop(key, None)
            path = self._path(key)
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    expires = datetime.fromisoformat(payload["expires"])
                    if expires > datetime.now(UTC):
                        candles = [Candle.model_validate(item) for item in payload["candles"]]
                        self._memory[key] = (expires, candles)
                        self.statistics.hits += 1
                        return list(candles)
                    path.unlink(missing_ok=True)
                except (OSError, ValueError, KeyError):
                    path.unlink(missing_ok=True)
            self.statistics.misses += 1
            return None

    async def set(self, key: str, candles: list[Candle], ttl_seconds: int) -> None:
        expires = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        async with self._lock:
            self._memory[key] = (expires, list(candles))
            self._memory.move_to_end(key)
            while len(self._memory) > self.max_entries:
                self._memory.popitem(last=False)
                self.statistics.evictions += 1
            payload = {"expires": expires.isoformat(), "candles": [item.model_dump(mode="json") for item in candles]}
            self._path(key).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            self.statistics.writes += 1
            self.statistics.last_write_at = datetime.now(UTC)

    async def invalidate(self, prefix: str = "") -> None:
        async with self._lock:
            keys = [key for key in self._memory if key.startswith(prefix)]
            for key in keys:
                self._memory.pop(key, None)
            for path in self.directory.glob("*.json"):
                if not prefix or path.stem.startswith(self._safe(prefix)):
                    path.unlink(missing_ok=True)
            self.statistics.invalidations += 1

    def _path(self, key: str) -> Path:
        return self.directory / f"{self._safe(key)}.json"

    @staticmethod
    def _safe(key: str) -> str:
        return "".join(character if character.isalnum() else "_" for character in key)
