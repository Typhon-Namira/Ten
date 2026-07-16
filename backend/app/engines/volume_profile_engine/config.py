from pydantic import BaseModel, Field


class VolumeProfileConfig(BaseModel):
    bins: int = Field(default=24, ge=4, le=200)
    value_area_percent: float = Field(default=0.70, gt=0, lt=1)
    high_volume_percentile: float = Field(default=0.75, gt=0, lt=1)

