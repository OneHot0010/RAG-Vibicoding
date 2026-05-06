"""Minimal MCP server entry point."""

from __future__ import annotations

from core.settings import SettingsError, load_settings
from observability.logger import get_logger


def main() -> int:
    """Load settings fail-fast before later server phases take over."""
    logger = get_logger(__name__)
    try:
        load_settings()
    except SettingsError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
