"""Helpers that prevent accidental secret disclosure."""


def mask_secret(value: str | None) -> str:
    """Return a log-safe representation of a secret."""

    if not value:
        return "<unset>"
    return f"{value[:3]}...{value[-2:]}" if len(value) > 6 else "***"

