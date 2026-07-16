from backend.app.engines.market_data_engine import Candle
from backend.app.engines.smc_engine import BaselineSMCAnalyzer, Bias


def test_detects_bullish_break(candles: list[Candle]) -> None:
    result = BaselineSMCAnalyzer().analyze(candles)
    assert result.bias == Bias.BULLISH
    assert result.structure_events[-1].kind == "BOS"


def test_short_input_is_neutral(candles: list[Candle]) -> None:
    result = BaselineSMCAnalyzer().analyze(candles[:2])
    assert result.bias == Bias.NEUTRAL
    assert result.observations

