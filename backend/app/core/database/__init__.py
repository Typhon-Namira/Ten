from .base import Base
from .schema import SCHEMA_HEAD_REVISION, prepare_database_schema
from .url import normalize_async_database_url
from typing import Any

__all__ = [
    "Base",
    "SCHEMA_HEAD_REVISION",
    "build_session_factory",
    "get_session",
    "normalize_async_database_url",
    "prepare_database_schema",
]


def __getattr__(name: str) -> Any:
    if name in {"build_session_factory", "get_session"}:
        from .session import build_session_factory, get_session

        return {"build_session_factory": build_session_factory, "get_session": get_session}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

