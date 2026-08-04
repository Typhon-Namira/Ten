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
    PrimaryScenarioPublicationRecord,
    PrimaryScenarioSelectionRecord,
    SignalEmailOutboxRecord,
)

logger = logging.getLogger(__name__)


class SignalEmailSender(Protocol):
    async def send(self, payload: dict[str, Any], recipient: str) -> str | None: ...


def primary_publication_ineligibility_reason(
    selection: Any,
    decision: Any,
    now: datetime,
) -> str | None:
    """Evaluate publication with Primary Scenario authority and guardrail evidence."""

    primary = selection.primary
    if selection.status.value != "SELECTED" or primary is None:
        return selection.rejection_reason or "primary_scenario_not_selected"
    if not selection.signal_eligible:
        return selection.rejection_reason or "primary_scenario_not_signal_eligible"
    lineage = getattr(decision, "source_lineage", None)
    if (
        lineage is None
        or lineage.primary_scenario_selection_id != selection.selection_id
    ):
        return "guardrail_decision_selection_mismatch"
    if getattr(getattr(decision, "mode", None), "value", None) != "live":
        return "guardrail_decision_not_live"
    if now >= primary.expiry or now >= decision.valid_until:
        return "primary_scenario_expired"
    if not decision.publication_eligible:
        return (
            decision.blockers[0].reason_code
            if decision.blockers
            else "guardrails_rejected"
        )
    if selection.authoritative_action.value not in {"BUY", "SELL"}:
        return "primary_scenario_direction_not_actionable"
    if primary.geometry is None:
        return "primary_scenario_geometry_missing"
    return None


def primary_scenario_email_outbox_values(
    selection: Any,
    decision: Any,
    recipient: str,
    now: datetime,
) -> dict[str, Any] | None:
    """Build an email solely from the authoritative Primary Scenario artifact."""

    if primary_publication_ineligibility_reason(selection, decision, now) is not None:
        return None
    primary = selection.primary
    assert primary is not None and primary.geometry is not None
    alternative = selection.alternative
    payload = {
        "symbol": selection.instrument,
        "signal_id": str(selection.selection_id),
        "primary_scenario_id": str(primary.candidate_id),
        "alternative_scenario_id": (
            str(alternative.candidate_id) if alternative is not None else None
        ),
        "cycle_id": str(selection.cycle_id),
        "direction": selection.authoritative_action.value,
        "primary_scenario_score": primary.final_scenario_score,
        "scenario_type": primary.scenario_type,
        "market_cutoff": selection.market_cutoff.isoformat(),
        "market_time": selection.market_cutoff.isoformat(),
        "reference_price": primary.reference_price,
        "expected_path": primary.deterministically_validated_path,
        "entry_type": primary.entry_type.value,
        "entry_zone": primary.geometry.entry_zone.model_dump(mode="json"),
        "entry": primary.geometry.entry,
        "stop_loss": primary.geometry.stop_loss,
        "take_profit": primary.geometry.take_profit,
        "risk_reward": primary.geometry.risk_reward_ratio,
        "invalidation": primary.geometry.reason,
        "expires_at": primary.expiry.isoformat(),
        "supporting_evidence": primary.supporting_evidence_ids[:5],
        "alternative_summary": (
            {
                "direction": alternative.direction.value,
                "scenario_type": alternative.scenario_type,
                "score": alternative.final_scenario_score,
            }
            if alternative is not None
            else None
        ),
        "decision_id": str(decision.decision_id),
        "guardrail_status": "APPROVED",
        "publication_status": "ELIGIBLE",
        "blockers": [
            item.reason_code for item in (*decision.blockers, *decision.warnings)
        ],
    }
    deduplication_key = sha256(
        f"{selection.selection_id}|{primary.candidate_id}|{recipient}".encode()
    ).hexdigest()
    return {
        "id": uuid5(NAMESPACE_URL, f"ten:primary-scenario-email:{deduplication_key}"),
        "signal_id": selection.selection_id,
        "primary_scenario_id": primary.candidate_id,
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

    async def evaluate_primary_scenario(
        self,
        selection: Any,
        decision: Any,
        recipient: str,
        *,
        email_enabled: bool,
        now: datetime | None = None,
    ) -> bool:
        """Persist evaluation and outbox atomically; retries repair missing outbox."""

        evaluated_at = now or datetime.now(UTC)
        reason = primary_publication_ineligibility_reason(
            selection, decision, evaluated_at
        )
        publication_status = "ELIGIBLE" if reason is None else "INELIGIBLE"
        email_values = (
            primary_scenario_email_outbox_values(
                selection, decision, recipient, evaluated_at
            )
            if reason is None and email_enabled
            else None
        )
        email_status = (
            "ELIGIBLE"
            if email_values is not None
            else "NOT_ELIGIBLE"
        )
        email_reason = (
            "primary_scenario_email_eligible"
            if email_values is not None
            else reason or "email_disabled"
        )
        primary_id = (
            selection.primary.candidate_id if selection.primary is not None else None
        )
        audit_payload = {
            "selection_id": str(selection.selection_id),
            "primary_scenario_id": str(primary_id) if primary_id else None,
            "market_cutoff": selection.market_cutoff.isoformat(),
            "publication_status": publication_status,
            "publication_reason": reason or "all_publication_guardrails_passed",
            "email_status": email_status,
            "email_reason": email_reason,
            "outbox": (
                {
                    "id": str(email_values["id"]),
                    "signal_id": str(email_values["signal_id"]),
                    "deduplication_key": email_values["deduplication_key"],
                    "recipient": email_values["recipient"],
                    "payload": email_values["payload"],
                }
                if email_values is not None
                else None
            ),
        }
        inserted_outbox = None
        async with self.session_factory() as session, session.begin():
            await session.execute(
                insert(PrimaryScenarioPublicationRecord)
                .values(
                    selection_id=selection.selection_id,
                    primary_scenario_id=primary_id,
                    decision_id=decision.decision_id,
                    publication_status=publication_status,
                    publication_reason=(reason or "all_publication_guardrails_passed"),
                    email_status=email_status,
                    email_reason=email_reason,
                    payload=audit_payload,
                    evaluated_at=evaluated_at,
                )
                .on_conflict_do_nothing(index_elements=["selection_id"])
            )
            if email_values is not None:
                inserted_outbox = (
                    await session.execute(
                        insert(SignalEmailOutboxRecord)
                        .values(**email_values)
                        .on_conflict_do_nothing()
                        .returning(SignalEmailOutboxRecord.id)
                    )
                ).scalar_one_or_none()
                await session.execute(
                    update(PrimaryScenarioPublicationRecord)
                    .where(
                        PrimaryScenarioPublicationRecord.selection_id
                        == selection.selection_id
                    )
                    .values(
                        email_status="ENQUEUED",
                        email_reason=(
                            "email_outbox_created"
                            if inserted_outbox is not None
                            else "email_outbox_already_exists"
                        ),
                    )
                )
        context = {
            "selection_id": str(selection.selection_id),
            "scenario_id": str(primary_id) if primary_id else None,
            "cutoff": selection.market_cutoff.isoformat(),
            "decision_id": str(decision.decision_id),
            "reason_code": reason or "all_publication_guardrails_passed",
        }
        logger.info("PUBLICATION_EVALUATED", extra=context)
        logger.info(
            "PUBLICATION_ELIGIBLE" if reason is None else "PUBLICATION_INELIGIBLE",
            extra=context,
        )
        logger.info("EMAIL_EVALUATED", extra={**context, "reason_code": email_reason})
        logger.info(
            "EMAIL_ENQUEUED" if email_values is not None else "EMAIL_NOT_ELIGIBLE",
            extra={
                **context,
                "reason_code": (
                    "email_outbox_created"
                    if inserted_outbox is not None
                    else "email_outbox_already_exists"
                    if email_values is not None
                    else email_reason
                ),
            },
        )
        return inserted_outbox is not None

    async def reconcile_primary_scenario_publications(self) -> int:
        """Repair an eligible publication whose durable outbox is absent."""

        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(PrimaryScenarioPublicationRecord)
                    .outerjoin(
                        SignalEmailOutboxRecord,
                        SignalEmailOutboxRecord.primary_scenario_id
                        == PrimaryScenarioPublicationRecord.primary_scenario_id,
                    )
                    .where(
                        PrimaryScenarioPublicationRecord.publication_status
                        == "ELIGIBLE",
                        PrimaryScenarioPublicationRecord.email_status.in_(
                            ("ELIGIBLE", "ENQUEUED")
                        ),
                        SignalEmailOutboxRecord.id.is_(None),
                    )
                    .limit(500)
                )
            ).all()
        inserted = 0
        for publication in rows:
            outbox = publication.payload.get("outbox")
            if not isinstance(outbox, dict) or publication.primary_scenario_id is None:
                continue
            now = datetime.now(UTC)
            values = {
                "id": UUID(str(outbox["id"])),
                "signal_id": UUID(str(outbox["signal_id"])),
                "primary_scenario_id": publication.primary_scenario_id,
                "deduplication_key": outbox["deduplication_key"],
                "decision_id": publication.decision_id,
                "recipient": outbox["recipient"],
                "status": "PENDING",
                "payload": outbox["payload"],
                "attempt_count": 0,
                "next_retry_at": now,
                "created_at": now,
                "updated_at": now,
            }
            async with self.session_factory() as session, session.begin():
                created = (
                    await session.execute(
                        insert(SignalEmailOutboxRecord)
                        .values(**values)
                        .on_conflict_do_nothing()
                        .returning(SignalEmailOutboxRecord.id)
                    )
                ).scalar_one_or_none()
            inserted += int(created is not None)
            logger.info(
                "EMAIL_ENQUEUED",
                extra={
                    "selection_id": str(publication.selection_id),
                    "scenario_id": str(publication.primary_scenario_id),
                    "cutoff": publication.payload.get("market_cutoff"),
                    "reason_code": "missing_outbox_reconciled",
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
        recipient: str | None = None,
    ) -> None:
        self.repository = repository
        self.sender = sender
        self.enabled = enabled
        self.poll_seconds = poll_seconds
        self.max_attempts = max_attempts
        self.recipient = recipient
        self._next_reconciliation_at: datetime | None = None
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
                now = datetime.now(UTC)
                if (
                    self.recipient is not None
                    and (
                        self._next_reconciliation_at is None
                        or now >= self._next_reconciliation_at
                    )
                ):
                    reconciled = (
                        await self.repository.reconcile_primary_scenario_publications()
                    )
                    self._next_reconciliation_at = now + timedelta(seconds=60)
                    logger.info(
                        "signal_email.reconciliation.completed",
                        extra={
                            "recipient": self.recipient,
                            "newly_queued": reconciled,
                            "reconciled_at": now.isoformat(),
                        },
                    )
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
                "EMAIL_SEND_STARTED",
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
                "EMAIL_SENT",
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
                "EMAIL_FAILED",
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
