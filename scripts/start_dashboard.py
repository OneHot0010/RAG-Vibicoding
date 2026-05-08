"""Start the local Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "src" / "observability" / "dashboard" / "app.py"


def main(argv: list[str] | None = None) -> int:
    """Launch Streamlit with the dashboard app."""
    argv = argv or []
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is required. Install it with `pip install streamlit`.", file=sys.stderr)
        return 1

    command = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *argv]
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
