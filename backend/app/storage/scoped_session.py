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
from contextvars import ContextVar
import functools
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_current_session: ContextVar[AsyncSession | None] = ContextVar("_current_session", default=None)


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
        async with self._session_factory() as session:
            token = _current_session.set(session)
            try:
                return await method(self, *args, **kwargs)
            finally:
                _current_session.reset(token)

    return wrapper  # type: ignore[return-value]


def scoped_session_stream[G: Callable[..., AsyncIterator[Any]]](method: G) -> G:
    """Variant of `scoped_session` for async-generator methods (e.g. replay history streaming):
    the session stays open for the full iteration, not just until the first `yield`."""

    @functools.wraps(method)
    async def wrapper(self: ScopedSessionRepository, *args: object, **kwargs: object) -> AsyncIterator[object]:
        async with self._session_factory() as session:
            token = _current_session.set(session)
            try:
                async for item in method(self, *args, **kwargs):
                    yield item
            finally:
                _current_session.reset(token)

    return wrapper  # type: ignore[return-value]
