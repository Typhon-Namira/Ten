"""Shared logical identities for immutable analytical persistence."""

from datetime import datetime
from enum import Enum
from hashlib import sha256
import inspect
import json
import logging
from typing import Protocol, cast

logger = logging.getLogger(__name__)


class AnalyticalSnapshot(Protocol):
    @property
    def symbol(self) -> str: ...

    @property
    def timeframe(self) -> object: ...

    @property
    def analysis_timestamp(self) -> datetime: ...

    @property
    def configuration_version(self) -> str: ...

    def model_dump(self, *, mode: str = "python") -> dict[str, object]: ...


class AnalyticalDeterminismError(RuntimeError):
    """The same immutable analytical boundary produced different analytical content."""

    def __init__(
        self,
        *,
        entity_type: str,
        logical_boundary: tuple[str, ...],
        existing_id: object,
        incoming_id: object,
        existing_hash: str,
        incoming_hash: str,
    ) -> None:
        self.entity_type = entity_type
        self.logical_boundary = logical_boundary
        self.existing_id = existing_id
        self.incoming_id = incoming_id
        self.existing_hash = existing_hash
        self.incoming_hash = incoming_hash
        super().__init__(
            f"{entity_type} is non-deterministic at {logical_boundary}: "
            f"existing_id={existing_id}, incoming_id={incoming_id}, "
            f"existing_hash={existing_hash}, incoming_hash={incoming_hash}"
        )


def _value(value: object) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def analytical_snapshot_boundary(
    snapshot: AnalyticalSnapshot,
    *,
    include_processing_mode: bool,
) -> tuple[str, ...]:
    """Return the schema-level uniqueness boundary used by analytical snapshots."""
    values = (
        snapshot.symbol.replace("/", "").upper(),
        _value(snapshot.timeframe),
        snapshot.analysis_timestamp.isoformat(),
        snapshot.configuration_version,
    )
    if not include_processing_mode:
        return values
    processing_mode = getattr(snapshot, "processing_mode", None)
    if processing_mode is None:
        raise ValueError("processing_mode is required by this snapshot boundary")
    return (*values, _value(processing_mode))


def analytical_payload_hash(snapshot: AnalyticalSnapshot) -> str:
    """Hash analytical content while ignoring regenerated storage metadata."""
    payload = snapshot.model_dump(mode="json")
    for field in ("id", "snapshot_id", "created_at"):
        payload.pop(field, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(encoded).hexdigest()


def ensure_analytical_determinism(
    existing: AnalyticalSnapshot,
    incoming: AnalyticalSnapshot,
    *,
    entity_type: str,
    include_processing_mode: bool,
) -> None:
    existing_id = getattr(existing, "id", getattr(existing, "snapshot_id", None))
    incoming_id = getattr(incoming, "id", getattr(incoming, "snapshot_id", None))
    # A semantic ID is itself derived from the analytical inputs. Reprocessing can legitimately
    # alter operational metadata (repository mode, recovery state, incremental mode) without
    # creating a second immutable artifact; the already-persisted row remains authoritative.
    if existing_id == incoming_id:
        return
    existing_hash = analytical_payload_hash(existing)
    incoming_hash = analytical_payload_hash(incoming)
    if existing_hash == incoming_hash:
        return
    error = AnalyticalDeterminismError(
        entity_type=entity_type,
        logical_boundary=analytical_snapshot_boundary(
            incoming,
            include_processing_mode=include_processing_mode,
        ),
        existing_id=existing_id,
        incoming_id=incoming_id,
        existing_hash=existing_hash,
        incoming_hash=incoming_hash,
    )
    logger.error(
        "analytical.persistence.non_deterministic_duplicate",
        extra={
            "entity_type": entity_type,
            "logical_identity": error.logical_boundary,
            "existing_id": str(existing_id),
            "incoming_id": str(incoming_id),
            "existing_payload_hash": existing_hash,
            "incoming_payload_hash": incoming_hash,
            "constraint_name": None,
            "exception_class": type(error).__name__,
        },
    )


def returned_identity(result: object, fallback: object) -> object | None:
    """Read INSERT..RETURNING while keeping lightweight repository fakes compatible."""
    reader = getattr(result, "scalar_one_or_none", None)
    if reader is None or inspect.iscoroutinefunction(reader):
        return fallback
    return cast(object | None, reader())
