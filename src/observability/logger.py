"""Minimal stderr logger used before observability is fully implemented."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "rag_vibecoding") -> logging.Logger:
    """Return a logger configured to write human-readable messages to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
