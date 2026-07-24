from datetime import UTC, datetime
from types import MethodType, SimpleNamespace

import pytest

from backend.app.integration.service import FullSystemIntegrationService
from backend.app.integration.service import _is_storage_exhausted


class DiskFullError(RuntimeError):
    pass


def test_disk_full_is_typed_across_wrapped_database_exceptions() -> None:
    root = DiskFullError('could not extend file "base/1/2": No space left on device')
    wrapped = RuntimeError("snapshot persistence failed")
    wrapped.__cause__ = root
    assert _is_storage_exhausted(wrapped)


def test_normal_database_failure_does_not_open_storage_circuit() -> None:
    assert not _is_storage_exhausted(RuntimeError("connection reset"))


@pytest.mark.asyncio
async def test_disk_full_opens_circuit_and_prevents_retry_write_storm() -> None:
    class Repository:
        def __init__(self) -> None:
            self.pending_calls = 0
            self.fail_calls = 0
            self.fail_record_storage_error = False

        async def pending(self, *_: object) -> list[SimpleNamespace]:
            self.pending_calls += 1
            return [SimpleNamespace(outbox_id="one", envelope=object())]

        async def fail(self, *_: object) -> None:
            self.fail_calls += 1
            if self.fail_record_storage_error:
                raise DiskFullError("No space left while recording failure")

    repository = Repository()
    service = object.__new__(FullSystemIntegrationService)
    service.repository = repository
    service.clock = lambda: datetime.now(UTC)
    service.config = SimpleNamespace(limits=SimpleNamespace(outbox_batch_size=20))
    service.failures = 0
    service.last_batch_failures = 0
    service.storage_exhausted_until = None

    async def fail_process(self: object, _: object) -> None:
        raise DiskFullError("No space left on device")

    service.process = MethodType(fail_process, service)

    assert await service.process_outbox_once() == 1
    assert service.storage_exhausted_until is not None
    assert await service.process_outbox_once() == 0
    assert repository.pending_calls == 1
    assert repository.fail_calls == 1

    service.storage_exhausted_until = None
    repository.fail_record_storage_error = True
    assert await service.process_outbox_once() == 1
    assert service.storage_exhausted_until is not None
