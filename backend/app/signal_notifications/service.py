from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
import logging
import smtplib
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import SignalEmailOutboxRecord

logger = logging.getLogger(__name__)


class SignalEmailSender(Protocol):
    async def send(self, payload: dict[str, Any], recipient: str) -> str | None: ...


def render_signal_email(payload: dict[str, Any]) -> tuple[str, str]:
    if payload.get("primary_scenario_id"):
        score = float(payload["primary_scenario_score"])
        subject = (
            f"TEN Primary Scenario · {payload['symbol']} {payload['direction']} · "
            f"{score:.0f}% · M15"
        )
        path = "\n".join(
            f"  {item}" for item in payload.get("expected_path") or ()
        )
        alternative = payload.get("alternative_summary") or {}
        primary_fields = (
            ("Instrument", payload["symbol"]),
            ("Market cutoff", payload.get("market_cutoff")),
            ("Direction", payload["direction"]),
            ("Scenario type", payload.get("scenario_type")),
            ("Primary Scenario score", f"{score:.1f}%"),
            ("Calibration", payload.get("calibration_status", "Pending")),
            ("Reference price", payload.get("reference_price")),
            ("Entry type", payload.get("entry_type")),
            ("Entry zone", payload.get("entry_zone")),
            ("Entry", payload.get("entry")),
            ("Stop Loss", payload.get("stop_loss")),
            ("Take Profit", payload.get("take_profit")),
            ("Risk/Reward", payload.get("risk_reward")),
            ("Invalidation", payload.get("invalidation")),
            ("Expiry", payload.get("expires_at")),
            ("Supporting evidence", ", ".join(payload.get("supporting_evidence") or ())),
            (
                "Alternative Scenario",
                (
                    f"{alternative.get('direction')} {alternative.get('scenario_type')} "
                    f"· {alternative.get('score')}%"
                    if alternative
                    else "Unavailable"
                ),
            ),
            ("Scenario ID", payload.get("primary_scenario_id")),
            ("Signal ID", payload.get("signal_id")),
        )
        body = "\n".join(
            [
                "TEN PRIMARY MARKET SCENARIO",
                "",
                "Expected path:",
                path,
                "",
                *[f"{label}: {value}" for label, value in primary_fields],
                "",
                "Analytical Intelligence Only",
                "No Broker Execution",
            ]
        )
        return subject, body
    raise ValueError("signal email requires an authoritative Primary Scenario")

class SmtpSignalEmailSender:
    def __init__(
        self,
        host: str,
        port: int,
        username: str | None,
        password: str | None,
        use_tls: bool,
        sender: str,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.sender = sender

    async def send(self, payload: dict[str, Any], recipient: str) -> str | None:
        subject, body = render_signal_email(payload)

        def send_sync() -> str | None:
            message = EmailMessage()
            message["From"] = self.sender
            message["To"] = recipient
            message["Subject"] = subject
            message["Message-ID"] = f"<ten-signal-{payload['signal_id']}@ten.local>"
            message.set_content(body)
            with smtplib.SMTP(self.host, self.port, timeout=30) as client:
                if self.use_tls:
                    client.starttls()
                if self.username:
                    client.login(self.username, self.password or "")
                response = client.send_message(message)
            return None if not response else str(response)

        return await asyncio.to_thread(send_sync)


class SignalEmailOutboxRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def claim(self, *, limit: int, now: datetime) -> tuple[SignalEmailOutboxRecord, ...]:
        async with self.session_factory() as session, session.begin():
            query = (
                select(SignalEmailOutboxRecord)
                .where(
                    SignalEmailOutboxRecord.next_retry_at <= now,
                    or_(
                        SignalEmailOutboxRecord.status.in_(("PENDING", "FAILED")),
                        (
                            (SignalEmailOutboxRecord.status == "PROCESSING")
                            & (SignalEmailOutboxRecord.processing_started_at < now - timedelta(minutes=5))
                        ),
                    ),
                )
                .order_by(SignalEmailOutboxRecord.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            records = tuple((await session.scalars(query)).all())
            for record in records:
                record.status = "PROCESSING"
                record.attempt_count += 1
                record.processing_started_at = now
                record.updated_at = now
            return records

    async def mark_sent(self, event_id: UUID, message_id: str | None, now: datetime) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                update(SignalEmailOutboxRecord)
                .where(
                    SignalEmailOutboxRecord.id == event_id,
                    SignalEmailOutboxRecord.status == "PROCESSING",
                )
                .values(
                    status="SENT",
                    provider_message_id=message_id,
                    sent_at=now,
                    updated_at=now,
                    last_error=None,
                )
            )

    async def mark_failed(
        self,
        event_id: UUID,
        *,
        attempt_count: int,
        error: str,
        next_retry_at: datetime,
        terminal: bool,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                update(SignalEmailOutboxRecord)
                .where(
                    SignalEmailOutboxRecord.id == event_id,
                    SignalEmailOutboxRecord.status == "PROCESSING",
                )
                .values(
                    status="FAILED",
                    attempt_count=attempt_count,
                    next_retry_at=next_retry_at,
                    processing_started_at=None,
                    last_error=error[:1000],
                    updated_at=datetime.now(UTC),
                )
            )
        if terminal:
            logger.error("signal_email.delivery.exhausted", extra={"event_id": str(event_id)})


class SignalEmailWorker:
    def __init__(
        self,
        repository: SignalEmailOutboxRepository,
        sender: SignalEmailSender,
        *,
        enabled: bool,
        poll_seconds: float,
        max_attempts: int,
    ) -> None:
        self.repository = repository
        self.sender = sender
        self.enabled = enabled
        self.poll_seconds = poll_seconds
        self.max_attempts = max_attempts
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        await self._task
        self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for event in await self.repository.claim(limit=10, now=datetime.now(UTC)):
                    await self._deliver(event)
            except Exception:
                logger.exception("signal_email.worker.failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def _deliver(self, event: SignalEmailOutboxRecord) -> None:
        attempt = event.attempt_count
        try:
            message_id = await self.sender.send(event.payload, event.recipient)
            await self.repository.mark_sent(event.id, message_id, datetime.now(UTC))
            logger.info(
                "signal_email.sent",
                extra={"event_id": str(event.id), "signal_id": str(event.signal_id)},
            )
        except Exception as exc:
            terminal = attempt >= self.max_attempts
            delay = min(3600, 30 * (2 ** (attempt - 1)))
            await self.repository.mark_failed(
                event.id,
                attempt_count=attempt,
                error=type(exc).__name__,
                next_retry_at=(
                    datetime.max.replace(tzinfo=UTC)
                    if terminal
                    else datetime.now(UTC) + timedelta(seconds=delay)
                ),
                terminal=terminal,
            )
