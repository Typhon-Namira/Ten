from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from backend.app.engines.market_data_engine import Candle, Timeframe

from .models import InstitutionalFlowEvidence, SessionType


@dataclass(frozen=True)
class InstitutionalFlowContext:
    candles: tuple[Candle, ...]
    evidence: tuple[InstitutionalFlowEvidence, ...] = ()
    session: SessionType = SessionType.UNKNOWN
    analysis_boundary: datetime | None = None
    upstream_versions: tuple[tuple[str, str], ...] = ()


class SMCEvidenceReader(Protocol):
    async def institutional_flow_evidence(self, symbol: str, timeframe: Timeframe, at: datetime) -> tuple[InstitutionalFlowEvidence, ...]: ...


class LiquidityEvidenceReader(Protocol):
    async def institutional_flow_evidence(self, symbol: str, timeframe: Timeframe, at: datetime) -> tuple[InstitutionalFlowEvidence, ...]: ...


class VolumeProfileEvidenceReader(Protocol):
    async def institutional_flow_evidence(self, symbol: str, timeframe: Timeframe, at: datetime) -> tuple[InstitutionalFlowEvidence, ...]: ...
