"""Application logging configuration."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent, container-friendly application logs."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # httpx logs each request's full URL (including query-string API keys, e.g. FMP/TwelveData's
    # `apikey=...`) at INFO level via its own logger, independent of anything this application
    # code does. At TEN's default INFO log level that would leak every keyed provider's secret
    # into stdout on every request — every engine's own request logging already redacts secrets
    # deliberately (see economic_calendar_engine/providers.py, market_data_engine/adapters.py);
    # httpx's built-in instrumentation must be held to the same bar.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

