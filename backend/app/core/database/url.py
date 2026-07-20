"""Database URL normalization for async SQLAlchemy runtimes."""


def normalize_async_database_url(url: str) -> str:
    """Convert Railway/PostgreSQL URLs to SQLAlchemy's asyncpg dialect."""
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return f"postgresql+asyncpg://{url.removeprefix('postgresql://')}"
    if url.startswith("postgres://"):
        return f"postgresql+asyncpg://{url.removeprefix('postgres://')}"
    return url
