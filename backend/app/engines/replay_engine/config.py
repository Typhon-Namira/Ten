from pydantic import BaseModel, Field


class ReplayConfig(BaseModel):
    enabled: bool
    clock_mode: str
    checkpoint_interval: int = Field(ge=1)
