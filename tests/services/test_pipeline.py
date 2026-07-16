import asyncio

from backend.app.engines.ai_scoring_engine import AIScoringEngine, ScoringContext, SignalScore
from backend.app.engines.ai_scoring_engine.models import ScoredDirection
from backend.app.engines.market_data_engine import Candle
from backend.app.engines.signal_engine import Direction
from backend.app.events import DashboardUpdated, MarketDataReady, SignalGenerated
from backend.app.services import AnalysisPipeline


class DeterministicScorer(AIScoringEngine):
    async def score(self, context: ScoringContext) -> SignalScore:
        assert context.features["smc"] and context.features["institutional_flow"]
        assert context.market_structure == {}
        return SignalScore(confidence=.78, direction=ScoredDirection.LONG, quality_score=80, reasoning=["Structured confluence"], model="test", prompt_version="test_v1")


def test_complete_pipeline_produces_scenario(candles: list[Candle]) -> None:
    pipeline = AnalysisPipeline(DeterministicScorer())
    scenario = asyncio.run(pipeline.analyze(candles, [], now=candles[-1].timestamp))
    assert scenario.direction == Direction.LONG
    assert scenario.symbol == "XAU/USD"
    assert scenario.reasoning
    assert scenario.explanation.confidence_breakdown
    event_types = [type(event) for event in pipeline.manager.event_bus.history()]
    assert event_types[0] is MarketDataReady
    assert SignalGenerated in event_types
    assert event_types[-1] is DashboardUpdated
