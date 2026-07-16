"""Extension ports for provider and future engine plugins."""

from abc import abstractmethod
from typing import Any

from backend.app.features import FeatureSnapshot

from .base import Plugin


class AIProviderPlugin(Plugin):
    @abstractmethod
    async def complete(self, features: FeatureSnapshot) -> dict[str, Any]:
        """Assess structured features; raw charts are outside this contract."""


class MarketDataProviderPlugin(Plugin):
    @abstractmethod
    async def fetch(self, symbol: str, timeframe: str, limit: int) -> list[dict[str, Any]]:
        """Fetch provider-specific records for normalization."""


class AnalysisEnginePlugin(Plugin):
    @abstractmethod
    async def execute(self, features: FeatureSnapshot) -> dict[str, Any]:
        """Produce structured features without invoking another engine."""


class NotificationProviderPlugin(Plugin):
    @abstractmethod
    async def notify(self, event_name: str, payload: dict[str, Any]) -> None:
        """Deliver a future outbound notification."""
