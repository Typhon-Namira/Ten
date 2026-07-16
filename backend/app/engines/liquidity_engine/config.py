from pydantic import BaseModel, Field


class LiquidityConfig(BaseModel):
    equal_level_tolerance: float = Field(default=0.001, gt=0)
    max_levels: int = Field(default=20, ge=1)

