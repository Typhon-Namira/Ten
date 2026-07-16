from backend.app.engines.ai_scoring_engine.models import ScoredDirection, SignalScore
from backend.app.engines.economic_calendar_engine.models import NewsRiskResult
from backend.app.engines.institutional_flow_engine.models import FlowBias, FlowScore
from backend.app.engines.liquidity_engine.models import LiquidityResult
from backend.app.engines.signal_engine import BaselineSignalEngine, Direction, SignalInputs
from backend.app.engines.smc_engine.models import Bias, SMCResult
from backend.app.engines.volume_profile_engine.models import VolumeProfileResult


def _inputs(no_trade: bool = False) -> SignalInputs:
    return SignalInputs(
        symbol="XAU/USD", timeframe="M15", current_price=2650, average_true_range=5,
        smc=SMCResult(bias=Bias.BULLISH), liquidity=LiquidityResult(),
        flow=FlowScore(score=.4, bias=FlowBias.BUYING, volume_pressure=.4, price_acceleration=.2, delta_estimate=.4, absorption_probability=.1),
        volume_profile=VolumeProfileResult(poc=2648, vah=2654, val=2642), news_risk=NewsRiskResult(no_trade=no_trade),
        ai_score=SignalScore(confidence=.8, direction=ScoredDirection.LONG, quality_score=82, reasoning=["Confluence"], model="test", prompt_version="v1"),
    )


def test_signal_engine_builds_non_executable_scenario() -> None:
    signal = BaselineSignalEngine().analyze(_inputs())
    assert signal.direction == Direction.LONG
    assert signal.stop_loss < signal.entry_zone[0] < signal.take_profit


def test_news_risk_forces_neutral() -> None:
    signal = BaselineSignalEngine().analyze(_inputs(no_trade=True))
    assert signal.direction == Direction.NEUTRAL
    assert signal.confidence == 0

