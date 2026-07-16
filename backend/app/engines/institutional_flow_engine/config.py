from pydantic import BaseModel, Field


class InstitutionalFlowConfig(BaseModel):
    acceleration_weight: float = Field(default=0.35, ge=0, le=1)
    volume_weight: float = Field(default=0.45, ge=0, le=1)
    close_location_weight: float = Field(default=0.20, ge=0, le=1)

