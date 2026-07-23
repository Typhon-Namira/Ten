"""Per-call AsyncSession scoping for SqlAlchemy repositories.

Root cause this fixes: every engine's repository previously held ONE `AsyncSession` for the
entire process lifetime (constructed once in `main.py`'s startup and stored on `app.state`).
`AsyncSession` is explicitly documented as unsafe for concurrent use — two coroutines issuing
statements on the same session at once raise exactly the production errors this was built to
eliminate: `InvalidRequestError: This session is provisioning a new connection; concurrent
operations are not permitted` and `...is in 'prepared' state; no further SQL can be emitted`.
A shared session is trivial to hit in production because the SAME engine's repository is read
by both the background integration pipeline (`_run()`) and concurrent dashboard API requests
(`/system/market-intelligence`, `/pipeline/stages/latest`, etc.) — nothing serialized access
between them.

The fix: repositories no longer store a `Session` at all — only a `session_factory`
(`async_sessionmaker`). `@scoped_session` wraps each public method so that call opens exactly
one fresh `AsyncSession` for its own duration (via a context-local variable, safe under
`asyncio.gather`/concurrent tasks because each `asyncio.Task` gets its own copy of the current
`contextvars.Context`), and the session is closed — rolling back anything uncommitted — the
moment that call returns, raises, or (for the two streaming replay sources) the generator is
exhausted. No session object ever survives past the one operation that opened it, and no two
concurrent calls can ever observe each other's session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from collections import deque
from contextvars import ContextVar
import functools
from time import perf_counter
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_current_session: ContextVar[AsyncSession | None] = ContextVar("_current_session", default=None)


class SessionRuntimeMetrics:
    def __init__(self) -> None:
        self.opened_total = 0
        self.completed_total = 0
        self.failed_total = 0
        self.active = 0
        self.peak_active = 0
        self.durations_ms: deque[float] = deque(maxlen=500)

    def started(self) -> float:
        self.opened_total += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        return perf_counter()

    def finished(self, started: float, *, failed: bool) -> None:
        self.active = max(0, self.active - 1)
        self.failed_total += int(failed)
        self.completed_total += int(not failed)
        self.durations_ms.append((perf_counter() - started) * 1000)

    def snapshot(self) -> dict[str, int | float]:
        ordered = sorted(self.durations_ms)
        p95 = ordered[max(0, round(0.95 * len(ordered)) - 1)] if ordered else 0.0
        return {
            "opened_total": self.opened_total,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
            "active": self.active,
            "peak_active": self.peak_active,
            "average_duration_ms": sum(self.durations_ms) / len(self.durations_ms) if self.durations_ms else 0.0,
            "p95_duration_ms": p95,
        }


session_runtime_metrics = SessionRuntimeMetrics()


class ScopedSessionRepository:
    """Base for SqlAlchemy repositories. Stores only a session factory — never a Session."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @property
    def session(self) -> AsyncSession:
        """The session bound to the currently-executing `@scoped_session`-wrapped call.

        Every existing repository method body already reads `self.session` — that code is
        untouched by this fix; only session *ownership* changed, not how methods use it.
        """
        session = _current_session.get()
        if session is None:
            raise RuntimeError(f"{type(self).__name__}.session accessed outside a @scoped_session-wrapped call")
        return session


def scoped_session[F: Callable[..., Awaitable[Any]]](method: F) -> F:
    """Give one method call its own AsyncSession, opened on entry and closed on exit/error."""

    @functools.wraps(method)
    async def wrapper(self: ScopedSessionRepository, *args: object, **kwargs: object) -> object:
        started = session_runtime_metrics.started()
        failed = False
        try:
            async with self._session_factory() as session:
                token = _current_session.set(session)
                try:
                    return await method(self, *args, **kwargs)
                finally:
                    _current_session.reset(token)
        except Exception:
            failed = True
            raise
        finally:
            session_runtime_metrics.finished(started, failed=failed)

    return wrapper  # type: ignore[return-value]


def scoped_session_stream[G: Callable[..., AsyncIterator[Any]]](method: G) -> G:
    """Variant of `scoped_session` for async-generator methods (e.g. replay history streaming):
    the session stays open for the full iteration, not just until the first `yield`."""

    @functools.wraps(method)
    async def wrapper(self: ScopedSessionRepository, *args: object, **kwargs: object) -> AsyncIterator[object]:
        started = session_runtime_metrics.started()
        failed = False
        try:
            async with self._session_factory() as session:
                token = _current_session.set(session)
                try:
                    async for item in method(self, *args, **kwargs):
                        yield item
                finally:
                    _current_session.reset(token)
        except Exception:
            failed = True
            raise
        finally:
            session_runtime_metrics.finished(started, failed=failed)

    return wrapper  # type: ignore[return-value]
