"""TEN 2.0 future-market application service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .models import ForecastPerformance, FutureMarketForecast
from .provider import BootstrapScenarioProvider, FutureMarketProvider
from .repository import FutureMarketRepository


class FutureMarketService:
    def __init__(
        self,
        repository: FutureMarketRepository,
        provider: FutureMarketProvider | None = None,
        *,
        simulation_repository: Any | None = None,
    ) -> None:
        self.repository = repository
        self.provider = provider or BootstrapScenarioProvider()
        self.simulation_repository = simulation_repository

    async def latest(self, instrument: str) -> FutureMarketForecast | None:
        """Return a fresh 30m forecast, bootstrapping from legacy scenario state if needed."""
        existing = await self.repository.latest(instrument)
        selection = (
            await self.simulation_repository.latest(instrument)
            if self.simulation_repository is not None
            else None
        )
        if selection is None:
            return existing

        market_cutoff = selection.market_cutoff
        if existing is not None and existing.market_cutoff >= market_cutoff:
            return existing

        candidates = await self.simulation_repository.candidates(selection.simulation_cycle_id)
        if not candidates:
            return existing
        primary = selection.primary or candidates[0]
        reference_price = float(primary.reference_price)
        forecast = await self.provider.forecast(
            instrument=instrument,
            market_cutoff=market_cutoff,
            generated_at=datetime.now(UTC),
            reference_price=reference_price,
            candidates=candidates,
        )
        return await self.repository.save(forecast)

    async def history(self, instrument: str, limit: int = 100) -> tuple[FutureMarketForecast, ...]:
        await self.latest(instrument)
        return await self.repository.history(instrument, limit)

    async def opportunities(self, instrument: str):
        forecast = await self.latest(instrument)
        return forecast.opportunities if forecast is not None else ()

    async def performance(self, instrument: str) -> ForecastPerformance:
        latest = await self.latest(instrument)
        return ForecastPerformance(
            instrument=instrument,
            forecasts_retained=await self.repository.count(instrument),
            provider=latest.provider if latest else None,
            model_name=latest.model_name if latest else None,
            model_version=latest.model_version if latest else None,
        )
