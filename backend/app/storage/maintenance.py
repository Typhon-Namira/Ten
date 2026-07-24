"""Bounded PostgreSQL retention and database-growth monitoring."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

# Identifiers are deliberately fixed in code. Policy rows may tune age/batch size but can never
# turn this worker into arbitrary SQL or delete immutable analytical evidence.
_CLEANABLE_RELATIONS = {
    "realtime_candles": "received_at",
    "pipeline_stage_history": "observed_at",
}


class StorageMaintenanceWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        interval_seconds: int = 3600,
    ) -> None:
        self.session_factory = session_factory
        self.interval_seconds = interval_seconds
        self.alert_bytes = int(os.getenv("TEN_DATABASE_SIZE_ALERT_BYTES", str(4 * 1024**3)))
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.last_database_bytes: int | None = None
        self.last_cleanup_at: datetime | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ten-storage-maintenance")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                logger.exception(
                    "storage.maintenance.failed",
                    extra={"exception_class": type(exc).__name__},
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass

    async def run_once(self) -> dict[str, int]:
        deleted: dict[str, int] = {}
        async with self.session_factory() as session:
            database_bytes = int(
                await session.scalar(text("SELECT pg_database_size(current_database())")) or 0
            )
            if database_bytes >= self.alert_bytes:
                logger.error(
                    "storage.database_size.alert",
                    extra={
                        "database_bytes": database_bytes,
                        "threshold_bytes": self.alert_bytes,
                        "reason": "database_size_threshold_exceeded",
                    },
                )
            policies = (
                await session.execute(
                    text(
                        """
                        SELECT relation_name, retention_days, cleanup_batch_size
                        FROM storage_retention_policies
                        WHERE protected = false
                        """
                    )
                )
            ).mappings().all()
            for policy in policies:
                relation = str(policy["relation_name"])
                timestamp_column = _CLEANABLE_RELATIONS.get(relation)
                if timestamp_column is None:
                    logger.warning(
                        "storage.retention.skipped",
                        extra={"relation": relation, "reason": "relation_not_allowlisted"},
                    )
                    continue
                # Relation and column are sourced only from the allowlist above. Values remain
                # bound parameters and every transaction deletes at most the configured batch.
                statement = text(
                    f"""
                    DELETE FROM {relation}
                    WHERE ctid IN (
                        SELECT ctid FROM {relation}
                        WHERE {timestamp_column} < now() - make_interval(days => :days)
                        ORDER BY {timestamp_column}
                        LIMIT :batch_size
                    )
                    """
                )
                result = await session.execute(
                    statement,
                    {
                        "days": int(policy["retention_days"]),
                        "batch_size": min(int(policy["cleanup_batch_size"]), 5000),
                    },
                )
                rowcount = getattr(result, "rowcount", 0)
                deleted[relation] = max(0, int(rowcount or 0))
            await session.commit()
        self.last_database_bytes = database_bytes
        self.last_cleanup_at = datetime.now(UTC)
        logger.info(
            "storage.maintenance.completed",
            extra={"database_bytes": database_bytes, "deleted": deleted},
        )
        return deleted
