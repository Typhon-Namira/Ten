from abc import ABC, abstractmethod

from backend.app.features import FeatureSnapshot

from .models import MarketRegimeResult


class MarketRegimeEngine(ABC):
    """Infrastructure contract only; no regime detection is implemented."""

    name = "market_regime"
    version = "1.0.0"
    compatibility_version = "1.0"

    @abstractmethod
    async def classify(self, features: FeatureSnapshot) -> MarketRegimeResult:
        """Classify a future feature snapshot."""
