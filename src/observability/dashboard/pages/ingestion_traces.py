"""Ingestion trace history page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from observability.dashboard.services.trace_service import TraceService


def ingestion_traces_model(
    trace_service: TraceService | None = None,
    *,
    trace_log_file: str | Path = "logs/traces.jsonl",
    selected_trace_id: str | None = None,
) -> dict[str, Any]:
    """Return display data for the ingestion traces page."""
    service = trace_service or TraceService(trace_log_file=trace_log_file)
    summaries = service.ingestion_summaries()
    selected = _select_summary(summaries, selected_trace_id)
    trace = service.get_trace(selected["trace_id"]) if selected is not None else None
    stage_rows = service.stage_rows(trace) if trace is not None else []
    return {
        "summaries": summaries,
        "selected_summary": selected,
        "trace": trace,
        "stage_rows": stage_rows,
        "waterfall_rows": _waterfall_rows(stage_rows),
        "stats": _trace_stats(trace),
    }


def render(
    st: Any | None = None,
    *,
    trace_service: TraceService | None = None,
    trace_log_file: str | Path = "logs/traces.jsonl",
) -> dict[str, Any]:
    """Render the ingestion traces page and return the underlying model for tests."""
    model = ingestion_traces_model(trace_service=trace_service, trace_log_file=trace_log_file)
    if st is None:
        return model

    st.title("Ingestion Traces")
    if not model["summaries"]:
        st.info("No ingestion traces found.")
        return model

    left, right = st.columns([1, 2])
    with left:
        st.subheader("History")
        st.dataframe(_summary_rows(model["summaries"]), use_container_width=True, hide_index=True)
        selected_id = st.selectbox("Trace", [summary["trace_id"] for summary in model["summaries"]])
        model = ingestion_traces_model(
            trace_service=trace_service,
            trace_log_file=trace_log_file,
            selected_trace_id=selected_id,
        )

    with right:
        _render_trace_detail(st, model)
    return model


def _render_trace_detail(st: Any, model: dict[str, Any]) -> None:
    summary = model["selected_summary"]
    if summary is None:
        st.info("Select a trace.")
        return

    st.subheader(Path(str(summary.get("source_path") or summary["trace_id"])).name)
    metrics = st.columns(4)
    metrics[0].metric("Status", summary["status"])
    metrics[1].metric("Elapsed ms", round(float(summary["total_elapsed_ms"]), 2))
    metrics[2].metric("Stages", summary["stage_count"])
    metrics[3].metric("Collection", summary["collection"] or "-")

    stats = model["stats"]
    st.write(
        {
            "trace_id": summary["trace_id"],
            "source_path": summary["source_path"],
            "started_at": summary["started_at"],
            "finished_at": summary["finished_at"],
            "chunk_count": stats["chunk_count"],
            "image_count": stats["image_count"],
            "skipped": stats["skipped"],
            "failed": stats["failed"],
        }
    )

    st.subheader("Stage Timing")
    if model["waterfall_rows"]:
        st.bar_chart(model["waterfall_rows"], x="stage", y="elapsed_ms")
    st.dataframe(_stage_table_rows(model["stage_rows"]), use_container_width=True, hide_index=True)

    st.subheader("Stage Details")
    for row in model["stage_rows"]:
        with st.expander(row["stage"], expanded=False):
            st.json(row["details"])


def _select_summary(summaries: list[dict[str, Any]], selected_trace_id: str | None) -> dict[str, Any] | None:
    if not summaries:
        return None
    if selected_trace_id:
        for summary in summaries:
            if summary["trace_id"] == selected_trace_id:
                return summary
    return summaries[0]


def _waterfall_rows(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"stage": row["stage"], "elapsed_ms": row["elapsed_ms"]}
        for row in stage_rows
        if float(row.get("elapsed_ms") or 0.0) > 0.0
    ]


def _trace_stats(trace: dict[str, Any] | None) -> dict[str, Any]:
    stats = {"chunk_count": 0, "image_count": 0, "skipped": False, "failed": False}
    if trace is None:
        return stats
    for stage in trace.get("stages", []):
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name") or "")
        data = stage.get("data") if isinstance(stage.get("data"), dict) else {}
        stats["failed"] = stats["failed"] or "failed" in name
        stats["skipped"] = stats["skipped"] or "skipped" in name or bool(data.get("skipped") is True)
        stats["chunk_count"] = max(stats["chunk_count"], _int_value(data.get("chunk_count"), data.get("count")))
        stats["image_count"] = max(stats["image_count"], _int_value(data.get("image_count")))
    return stats


def _int_value(*values: Any) -> int:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "trace_id": summary["trace_id"],
            "status": summary["status"],
            "source_path": summary["source_path"],
            "collection": summary["collection"],
            "elapsed_ms": round(float(summary["total_elapsed_ms"]), 3),
            "started_at": summary["started_at"],
        }
        for summary in summaries
    ]


def _stage_table_rows(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": row["index"],
            "stage": row["stage"],
            "elapsed_ms": row["elapsed_ms"],
            "method": row["method"],
            "provider": row["provider"],
        }
        for row in stage_rows
    ]


__all__ = ["ingestion_traces_model", "render"]
