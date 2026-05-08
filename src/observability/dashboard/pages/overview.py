"""System overview page for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services.config_service import ConfigService


def overview_model(settings_path: str = "config/settings.yaml", data_dir: str = "data") -> dict[str, Any]:
    """Return display data for the overview page."""
    return ConfigService(settings_path=settings_path, data_dir=data_dir).overview()


def render(st: Any | None = None, settings_path: str = "config/settings.yaml", data_dir: str = "data") -> dict[str, Any]:
    """Render the overview page and return the underlying model for tests."""
    model = overview_model(settings_path=settings_path, data_dir=data_dir)
    if st is None:
        return model

    st.title("System Overview")
    st.caption("RAG Vibecoding local knowledge hub")

    assets = model["assets"]
    columns = st.columns(4)
    columns[0].metric("Documents", assets["document_count"])
    columns[1].metric("Chunks", assets["chunk_count"])
    columns[2].metric("Images", assets["image_count"])
    columns[3].metric("Traces", assets["trace_count"])

    st.subheader("Components")
    for component in model["components"]:
        with st.expander(component["name"], expanded=True):
            st.write(
                {
                    "provider": component["provider"],
                    "model": component["model"],
                    **component["details"],
                }
            )

    st.subheader("Dashboard")
    st.write(model["dashboard"])
    return model
