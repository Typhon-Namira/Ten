"""Provider boundary for TEN 2.0 future-market forecasting."""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from .models import (
    FORECAST_HORIZON_SECONDS,
    FutureMarketForecast,
    FuturePathStage,
    MarketStateSummary,
    OpportunityWindow,
    PriceZone,
    ScenarioBranch,
    ScenarioDirection,
)


class FutureMarketProvider(Protocol):
    name: str
    model_name: str
    model_version: str

    async def forecast(
        self,
        *,
        instrument: str,
        market_cutoff,
        generated_at,
        reference_price: float,
        candidates: Sequence[object],
    ) -> FutureMarketForecast: ...


class BootstrapScenarioProvider:
    """Bridge the current deterministic scenario engine into the TEN 2.0 contract.

    This adapter is intentionally disposable.  Lightning replaces only this provider;
    the repository, API and dashboard contracts remain unchanged.
    """

    name = "bootstrap_scenario_provider"
    model_name = "ten-bootstrap-scenario-world-model"
    model_version = "2.0-bootstrap.1"

    async def forecast(
        self,
        *,
        instrument: str,
        market_cutoff,
        generated_at,
        reference_price: float,
        candidates: Sequence[object],
    ) -> FutureMarketForecast:
        usable = list(candidates[:3])
        raw_scores = [max(float(getattr(item, "final_scenario_score", 0.0)), 1.0) for item in usable]
        total = sum(raw_scores) or 1.0
        branches: list[ScenarioBranch] = []
        opportunities: list[OpportunityWindow] = []
        for rank, (candidate, raw_score) in enumerate(zip(usable, raw_scores, strict=True), start=1):
            probability = raw_score / total
            candidate_id = getattr(candidate, "candidate_id")
            direction = ScenarioDirection(getattr(candidate, "direction").value)
            expected_low = float(getattr(candidate, "expected_low"))
            expected_high = float(getattr(candidate, "expected_high"))
            close_low = float(getattr(candidate, "likely_close_low"))
            close_high = float(getattr(candidate, "likely_close_high"))
            stages = []
            source_stages = tuple(getattr(candidate, "path_sequence", ()) or ())
            for sequence, stage in enumerate(source_stages, start=1):
                timing = int(getattr(stage, "timing_seconds", sequence * 300) or sequence * 300)
                minute_to = min(30, max(1, round(timing / 60)))
                minute_from = 0 if sequence == 1 else min(29, max(0, minute_to - 5))
                area = getattr(stage, "expected_price_area", None)
                stages.append(
                    FuturePathStage(
                        sequence=sequence,
                        minute_from=minute_from,
                        minute_to=minute_to,
                        event=str(getattr(stage, "label", "market_transition")),
                        expected_price_area=(
                            PriceZone(low=float(area.low), high=float(area.high))
                            if area is not None
                            else None
                        ),
                        invalidation_condition=getattr(stage, "invalidation_condition", None),
                    )
                )
            branch = ScenarioBranch(
                scenario_id=candidate_id,
                scenario_type=str(getattr(candidate, "scenario_type")),
                direction=direction,
                probability=probability,
                expected_range=PriceZone(low=expected_low, high=expected_high),
                likely_close=PriceZone(low=close_low, high=close_high),
                path=tuple(stages) or (
                    FuturePathStage(
                        sequence=1,
                        minute_from=0,
                        minute_to=30,
                        event="scenario_unfolding",
                        expected_price_area=PriceZone(low=expected_low, high=expected_high),
                    ),
                ),
                invalidation=str(getattr(candidate, "rejection_reason", None) or "scenario structure invalidated"),
                rank=rank,
            )
            branches.append(branch)

            geometry = getattr(candidate, "geometry", None)
            entry_zone = getattr(candidate, "entry_zone", None)
            invalidation = getattr(candidate, "invalidation_level", None)
            targets = tuple(
                value
                for value in (
                    getattr(candidate, "primary_target", None),
                    getattr(candidate, "secondary_target", None),
                )
                if value is not None
            )
            if (
                direction in {ScenarioDirection.BULLISH, ScenarioDirection.BEARISH}
                and entry_zone is not None
                and invalidation is not None
                and targets
            ):
                trigger = str(getattr(candidate, "trigger_condition", "scenario trigger confirms"))
                opportunities.append(
                    OpportunityWindow(
                        opportunity_id=uuid5(
                            NAMESPACE_URL,
                            f"ten:v2:opportunity:{candidate_id}:{market_cutoff.isoformat()}",
                        ),
                        scenario_id=candidate_id,
                        direction=direction,
                        expected_from_minute=5,
                        expected_to_minute=15,
                        entry_zone=PriceZone(low=float(entry_zone.low), high=float(entry_zone.high)),
                        trigger_conditions=(trigger,),
                        invalidation_level=float(invalidation),
                        targets=tuple(float(item) for item in targets),
                        probability=probability,
                        quality=min(100.0, max(0.0, float(getattr(candidate, "final_scenario_score", 0.0)))),
                    )
                )

        dominant = branches[0].scenario_id if branches else None
        uncertainty = 1.0 - (branches[0].probability if branches else 0.0)
        regime = (
            branches[0].direction.value if branches else ScenarioDirection.INCONCLUSIVE.value
        )
        forecast_id = uuid5(
            NAMESPACE_URL,
            f"ten:v2:future-market:{instrument}:{market_cutoff.isoformat()}:{self.model_version}",
        )
        return FutureMarketForecast(
            forecast_id=forecast_id,
            instrument=instrument,
            generated_at=generated_at,
            market_cutoff=market_cutoff,
            expires_at=market_cutoff + timedelta(seconds=FORECAST_HORIZON_SECONDS),
            provider=self.name,
            model_name=self.model_name,
            model_version=self.model_version,
            market_state=MarketStateSummary(
                regime=regime,
                uncertainty=uncertainty,
                reference_price=reference_price,
            ),
            dominant_scenario_id=dominant,
            scenarios=tuple(branches),
            opportunities=tuple(opportunities),
        )
