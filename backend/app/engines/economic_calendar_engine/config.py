from pydantic import BaseModel, Field


class EconomicCalendarConfig(BaseModel):
    high_impact_pre_minutes: int = Field(default=30, ge=0)
    high_impact_post_minutes: int = Field(default=30, ge=0)
    medium_impact_pre_minutes: int = Field(default=15, ge=0)
    medium_impact_post_minutes: int = Field(default=15, ge=0)

