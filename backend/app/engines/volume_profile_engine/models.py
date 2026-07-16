from pydantic import BaseModel, Field


class PriceNode(BaseModel):
    price: float
    volume: float = Field(ge=0)
    kind: str


class VolumeProfileResult(BaseModel):
    poc: float | None = None
    vah: float | None = None
    val: float | None = None
    total_volume: float = Field(default=0, ge=0)
    nodes: list[PriceNode] = Field(default_factory=list)
    profile_type: str = "composite"
    observations: list[str] = Field(default_factory=list)

