import asyncio

import pytest
from pydantic import ValidationError

from backend.app.engines.market_data_engine import Candle, InMemoryMarketDataProvider, Timeframe


def test_in_memory_provider_filters_and_orders(candles: list[Candle]) -> None:
    provider = InMemoryMarketDataProvider(list(reversed(candles)))
    result = asyncio.run(provider.candles("XAU/USD", Timeframe.M15, limit=2))
    assert result == candles[-2:]


def test_candle_rejects_invalid_ohlc(candles: list[Candle]) -> None:
    invalid = candles[0].model_dump()
    invalid["high"] = 2600
    with pytest.raises(ValidationError):
        Candle.model_validate(invalid)
