from backend.app.engines.liquidity_engine import BaselineLiquidityAnalyzer
from backend.app.engines.market_data_engine import Candle


def test_maps_equal_highs(candles: list[Candle]) -> None:
    result = BaselineLiquidityAnalyzer().analyze(candles)
    assert any(level.touches >= 2 for level in result.levels)
    assert result.active_session == "london"


def test_empty_liquidity_input_is_valid() -> None:
    result = BaselineLiquidityAnalyzer().analyze([])
    assert result.levels == []
    assert result.observations

