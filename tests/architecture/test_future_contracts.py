import inspect

from backend.app.ai.memory import AIMemory
from backend.app.engines.market_regime_engine import MarketRegime, MarketRegimeEngine
from backend.app.engines.replay_engine import ReplayEngine


def test_future_engines_are_infrastructure_only() -> None:
    assert inspect.isabstract(MarketRegimeEngine)
    assert inspect.isabstract(ReplayEngine)
    assert inspect.isabstract(AIMemory)
    assert MarketRegime.UNKNOWN.value == "unknown"
