from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ReplayStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ReplayRequest(BaseModel):
    replay_id: UUID = Field(default_factory=uuid4)
    symbol: str
    timeframe: str
    start_at: datetime
    end_at: datetime


class ReplayCheckpoint(BaseModel):
    replay_id: UUID
    cursor_at: datetime
    processed_events: int = Field(ge=0)


class ReplayState(BaseModel):
    request: ReplayRequest
    status: ReplayStatus = ReplayStatus.CREATED
    checkpoint: ReplayCheckpoint | None = None
