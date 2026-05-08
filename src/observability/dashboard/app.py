"""Streamlit dashboard entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from observability.dashboard.pages import data_browser, ingestion_manager, ingestion_traces, overview


PageRenderer = Callable[[Any], None]


def page_registry() -> dict[str, PageRenderer]:
    """Return dashboard page renderers in navigation order."""
    return {
        "Overview": lambda st: overview.render(st),
        "Data Browser": lambda st: data_browser.render(st),
        "Ingestion Manager": lambda st: ingestion_manager.render(st),
        "Ingestion Traces": lambda st: ingestion_traces.render(st),
        "Query Traces": lambda st: _placeholder(st, "Query Traces"),
        "Evaluation": lambda st: _placeholder(st, "Evaluation"),
    }


def main() -> None:
    """Run the Streamlit dashboard."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Streamlit is required. Install it with `pip install streamlit`.") from exc

    st.set_page_config(page_title="RAG Vibecoding Dashboard", layout="wide")
    st.sidebar.title("RAG Vibecoding")
    pages = page_registry()
    selected = st.sidebar.radio("Navigate", list(pages), index=0)
    pages[selected](st)


def _placeholder(st: Any, title: str) -> None:
    st.title(title)
    st.info("This dashboard page will be implemented in a later milestone.")


if __name__ == "__main__":
    main()
