import inspect

from backend.app.ai.memory import AIMemory
from backend.app.engines.market_regime_engine import DominantRegime, MarketRegimeEngine
from backend.app.engines.replay_engine import ReplayEngine


def test_abstract_engine_contracts_remain_explicit() -> None:
    assert inspect.isabstract(MarketRegimeEngine)
    assert inspect.isabstract(ReplayEngine)
    assert inspect.isabstract(AIMemory)
    assert DominantRegime.UNCERTAIN.value == "uncertain"
