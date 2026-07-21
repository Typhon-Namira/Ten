import logging

from backend.app.core.logging import configure_logging


def test_configure_logging_suppresses_httpx_request_line_logging() -> None:
    """httpx logs each request's full URL — including query-string API keys like FMP's
    `apikey=...` or TwelveData's `apikey=...` — at INFO level via its own logger, independent of
    anything TEN's own code does. At TEN's default INFO log level that would leak every keyed
    provider's secret into stdout on every single request unless explicitly suppressed."""
    configure_logging("INFO")
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING
