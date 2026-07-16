from pydantic import BaseModel


class MarketRegimeConfig(BaseModel):
    compatibility_version: str
    enabled: bool
