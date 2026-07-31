from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from hashlib import sha256
import logging
import smtplib
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    PrimaryScenarioSelectionRecord,
    SignalDecisionRecord,
    SignalEmailOutboxRecord,
)

logger = logging.getLogger(__name__)


class SignalEmailSender(Protocol):
    async def send(self, payload: dict[str, Any], recipient: str) -> str | None: ...


def primary_email_outbox_values(
    decision: Any,
    recipient: str,
    now: datetime,
) -> dict[str, Any] | None:
    notification = decision.notification_context
    if not (
        getattr(getattr(decision, "mode", None), "value", None) == "live"
        and decision.publication_eligible
        and notification is not None
        and notification.get("primary_scenario_id") is not None
        and notification.get("direction") in {"BUY", "SELL"}
        and all(
            notification.get(field) is not None
            for field in ("entry", "stop_loss", "take_profit", "risk_reward")
        )
        and float(notification["risk_reward"]) > 0
        and float(notification.get("primary_scenario_score", 100))
        >= float(notification.get("email_threshold", 0))
        and decision.decided_at < decision.valid_until
        and now < decision.valid_until
    ):
        return None
    expires_at = notification.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    if isinstance(expires_at, datetime) and (
        decision.decided_at >= expires_at or now >= expires_at
    ):
        return None
    payload = {
        **notification,
        "symbol": decision.instrument,
        "market_time": decision.as_of.isoformat(),
        "decision_id": str(decision.decision_id),
        "guardrail_status": "APPROVED",
        "publication_status": "ELIGIBLE",
        "blockers": [
            item.reason_code for item in (*decision.blockers, *decision.warnings)
        ],
    }
    deduplication_key = sha256(
        "|".join(
            str(value)
            for value in (
                decision.instrument,
                notification.get("market_cutoff", decision.as_of.isoformat()),
                notification["primary_scenario_id"],
                notification["direction"],
                notification["entry"],
                notification["stop_loss"],
                notification["take_profit"],
            )
        ).encode()
    ).hexdigest()
    return {
        "id": uuid5(NAMESPACE_URL, f"ten:primary-scenario-email:{deduplication_key}"),
        "signal_id": UUID(str(notification["signal_id"])),
        "primary_scenario_id": UUID(str(notification["primary_scenario_id"])),
        "deduplication_key": deduplication_key,
        "decision_id": decision.decision_id,
        "recipient": recipient,
        "status": "PENDING",
        "payload": payload,
        "attempt_count": 0,
        "next_retry_at": now,
        "created_at": now,
        "updated_at": now,
    }


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
            message_id = (
                f"<ten-primary-scenario-{payload['primary_scenario_id']}@ten.local>"
            )
            message["Message-ID"] = message_id
            message.set_content(body)
            with smtplib.SMTP(self.host, self.port, timeout=30) as client:
                if self.use_tls:
                    client.starttls()
                if self.username:
                    client.login(self.username, self.password or "")
                response = client.send_message(message)
            if response:
                raise RuntimeError("SMTP rejected one or more recipients")
            return message_id

        return await asyncio.to_thread(send_sync)


class SignalEmailOutboxRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def enqueue_decision(self, decision: Any, recipient: str) -> bool:
        now = datetime.now(UTC)
        values = primary_email_outbox_values(decision, recipient, now)
        if values is None:
            return False
        async with self.session_factory() as session, session.begin():
            inserted = (
                await session.execute(
                    insert(SignalEmailOutboxRecord)
                    .values(**values)
                    .on_conflict_do_nothing()
                    .returning(SignalEmailOutboxRecord.id)
                )
            ).scalar_one_or_none()
        logger.info(
            "signal_email.triggered" if inserted is not None else "signal_email.duplicate",
            extra={
                "scenario_id": str(values["primary_scenario_id"]),
                "cutoff": values["payload"].get("market_cutoff"),
                "email_trigger_time": now.isoformat(),
                "recipient": recipient,
                "event_id": str(values["id"]),
                "delivery_state": "QUEUED" if inserted is not None else "ALREADY_QUEUED",
            },
        )
        return inserted is not None

    async def reconcile_eligible_decisions(
        self,
        recipient: str,
        *,
        limit: int = 500,
    ) -> int:
        from backend.app.engines.signal_decision_engine.models import SignalDecision

        async with self.session_factory() as session:
            records = tuple(
                (
                    await session.scalars(
                        select(SignalDecisionRecord)
                        .where(
                            SignalDecisionRecord.mode == "live",
                            SignalDecisionRecord.state == "eligible",
                        )
                        .order_by(SignalDecisionRecord.decided_at.desc())
                        .limit(limit)
                    )
                ).all()
            )
        inserted = 0
        for record in records:
            try:
                decision = SignalDecision.model_validate(record.payload)
                inserted += int(await self.enqueue_decision(decision, recipient))
            except Exception as exc:
                logger.warning(
                    "signal_email.reconciliation.skipped",
                    extra={
                        "decision_id": str(record.id),
                        "failure_reason": type(exc).__name__,
                    },
                )
        return inserted

    async def for_primary_scenario(
        self, primary_scenario_id: UUID
    ) -> SignalEmailOutboxRecord | None:
        async with self.session_factory() as session:
            return (
                await session.scalars(
                    select(SignalEmailOutboxRecord)
                    .where(
                        SignalEmailOutboxRecord.primary_scenario_id
                        == primary_scenario_id
                    )
                    .order_by(SignalEmailOutboxRecord.created_at.desc())
                    .limit(1)
                )
            ).first()

    async def delivery_summary(self) -> dict[str, int]:
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        SignalEmailOutboxRecord.status,
                        func.count(SignalEmailOutboxRecord.id),
                    ).group_by(SignalEmailOutboxRecord.status)
                )
            ).all()
            statuses = {str(status): int(count) for status, count in rows}
            eligible_scenarios = int(
                (
                    await session.scalar(
                        select(func.count(PrimaryScenarioSelectionRecord.selection_id))
                        .where(
                            PrimaryScenarioSelectionRecord.signal_eligible.is_(True)
                        )
                    )
                )
                or 0
            )
        return {
            "eligible_scenarios": eligible_scenarios,
            "triggered": sum(statuses.values()),
            "delivered": statuses.get("SENT", 0),
            "failed": statuses.get("FAILED", 0)
            + statuses.get("PERMANENTLY_FAILED", 0),
        }

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
                    status="PERMANENTLY_FAILED" if terminal else "FAILED",
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
            logger.info(
                "signal_email.sending",
                extra={
                    "event_id": str(event.id),
                    "scenario_id": str(event.primary_scenario_id),
                    "cutoff": event.payload.get("market_cutoff"),
                    "recipient": event.recipient,
                    "attempt": attempt,
                    "delivery_state": "SENDING",
                },
            )
            message_id = await self.sender.send(event.payload, event.recipient)
            await self.repository.mark_sent(event.id, message_id, datetime.now(UTC))
            logger.info(
                "signal_email.sent",
                extra={
                    "event_id": str(event.id),
                    "signal_id": str(event.signal_id),
                    "scenario_id": str(event.primary_scenario_id),
                    "cutoff": event.payload.get("market_cutoff"),
                    "recipient": event.recipient,
                    "provider_response": "accepted",
                    "message_id": message_id,
                    "delivery_state": "DELIVERED",
                },
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
            logger.warning(
                "signal_email.failed",
                extra={
                    "event_id": str(event.id),
                    "scenario_id": str(event.primary_scenario_id),
                    "cutoff": event.payload.get("market_cutoff"),
                    "recipient": event.recipient,
                    "success": False,
                    "failure_reason": type(exc).__name__,
                    "attempt": attempt,
                    "delivery_state": (
                        "PERMANENTLY_FAILED" if terminal else "RETRY_SCHEDULED"
                    ),
                },
            )
