from pydantic import BaseModel, Field


class SMCConfig(BaseModel):
    swing_window: int = Field(default=3, ge=1)
    minimum_displacement_ratio: float = Field(default=0.5, ge=0)
    equal_level_tolerance: float = Field(default=0.0005, gt=0)

