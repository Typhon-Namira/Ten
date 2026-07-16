from pydantic import BaseModel


class MarketDataConfig(BaseModel):
    symbol: str = "XAU/USD"
    default_lookback: int = 500
    request_timeout_seconds: float = 15.0

