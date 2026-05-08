"""Logging helpers for stderr diagnostics and JSON Lines traces."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any


DEFAULT_TRACE_LOG_FILE = Path("logs/traces.jsonl")


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


class JSONFormatter(logging.Formatter):
    """Format logging records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a compact JSON representation of a log record."""
        if isinstance(record.msg, dict):
            payload = dict(record.msg)
        else:
            payload = {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def get_trace_logger(
    log_file: str | Path = DEFAULT_TRACE_LOG_FILE,
    name: str = "rag_vibecoding.trace",
) -> logging.Logger:
    """Return a logger configured to append trace payloads as JSON Lines."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved = str(path.resolve())
    if not _has_file_handler(logger, resolved):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        handler.setLevel(logging.INFO)
        handler._rag_trace_log_file = resolved  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def write_trace(trace_dict: dict[str, Any], log_file: str | Path = DEFAULT_TRACE_LOG_FILE) -> None:
    """Append one serialized trace dictionary to the JSON Lines trace log."""
    if not isinstance(trace_dict, dict):
        raise ValueError("trace_dict must be a dict")
    get_trace_logger(log_file).info(trace_dict)


def _has_file_handler(logger: logging.Logger, resolved_path: str) -> bool:
    return any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "_rag_trace_log_file", None) == resolved_path
        for handler in logger.handlers
    )
