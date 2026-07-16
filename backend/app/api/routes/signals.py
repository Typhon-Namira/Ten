from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.dependencies import get_signal_repository
from backend.app.engines.signal_engine import Signal
from backend.app.services import SignalRepository

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("", response_model=list[Signal])
async def list_signals(repository: Annotated[SignalRepository, Depends(get_signal_repository)], limit: Annotated[int, Query(ge=1, le=250)] = 50) -> list[Signal]:
    """Return generated market scenarios newest first."""

    return await repository.list(limit)


@router.get("/latest", response_model=Signal)
async def latest_signal(repository: Annotated[SignalRepository, Depends(get_signal_repository)]) -> Signal:
    """Return the newest generated scenario."""

    signal = await repository.latest()
    if signal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No signal has been generated")
    return signal

