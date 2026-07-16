from backend.app.engines.market_data_engine import Candle
from backend.app.engines.volume_profile_engine import BaselineVolumeProfileAnalyzer


def test_profile_has_ordered_value_area(candles: list[Candle]) -> None:
    result = BaselineVolumeProfileAnalyzer().analyze(candles)
    assert result.poc is not None
    assert result.val is not None and result.vah is not None
    assert result.val <= result.vah
    assert result.total_volume == sum(item.volume for item in candles)

