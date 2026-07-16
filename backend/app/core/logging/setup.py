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

