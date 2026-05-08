"""Query trace history page."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services.trace_service import TraceService


def query_traces_model(
    trace_service: TraceService | None = None,
    *,
    trace_log_file: str = "logs/traces.jsonl",
    selected_trace_id: str | None = None,
    search: str = "",
) -> dict[str, Any]:
    """Return display data for the query traces page."""
    service = trace_service or TraceService(trace_log_file=trace_log_file)
    summaries = _filter_summaries(service.query_summaries(), search)
    selected = _select_summary(summaries, selected_trace_id)
    trace = service.get_trace(selected["trace_id"]) if selected is not None else None
    stage_rows = service.stage_rows(trace) if trace is not None else []
    return {
        "summaries": summaries,
        "selected_summary": selected,
        "trace": trace,
        "stage_rows": stage_rows,
        "waterfall_rows": _waterfall_rows(stage_rows),
        "stats": _query_stats(trace),
        "dense_rows": _route_rows(trace, route="dense"),
        "sparse_rows": _route_rows(trace, route="sparse"),
        "fusion_rows": _fusion_rows(trace),
        "rerank_rows": _rerank_rows(trace),
    }


def render(
    st: Any | None = None,
    *,
    trace_service: TraceService | None = None,
    trace_log_file: str = "logs/traces.jsonl",
) -> dict[str, Any]:
    """Render the query traces page and return the underlying model for tests."""
    model = query_traces_model(trace_service=trace_service, trace_log_file=trace_log_file)
    if st is None:
        return model

    st.title("Query Traces")
    search = st.text_input("Search query", value="")
    model = query_traces_model(trace_service=trace_service, trace_log_file=trace_log_file, search=search)
    if not model["summaries"]:
        st.info("No query traces found.")
        return model

    left, right = st.columns([1, 2])
    with left:
        st.subheader("History")
        st.dataframe(_summary_rows(model["summaries"]), use_container_width=True, hide_index=True)
        selected_id = st.selectbox("Trace", [summary["trace_id"] for summary in model["summaries"]])
        model = query_traces_model(
            trace_service=trace_service,
            trace_log_file=trace_log_file,
            selected_trace_id=selected_id,
            search=search,
        )

    with right:
        _render_trace_detail(st, model)
    return model


def _render_trace_detail(st: Any, model: dict[str, Any]) -> None:
    summary = model["selected_summary"]
    if summary is None:
        st.info("Select a trace.")
        return

    st.subheader(summary.get("query") or summary["trace_id"])
    metrics = st.columns(4)
    metrics[0].metric("Status", summary["status"])
    metrics[1].metric("Elapsed ms", round(float(summary["total_elapsed_ms"]), 2))
    metrics[2].metric("Results", model["stats"]["result_count"])
    metrics[3].metric("Rerank", "on" if model["stats"]["rerank_enabled"] else "off")

    st.write(
        {
            "trace_id": summary["trace_id"],
            "collection": summary["collection"],
            "started_at": summary["started_at"],
            "finished_at": summary["finished_at"],
            **model["stats"],
        }
    )

    st.subheader("Stage Timing")
    if model["waterfall_rows"]:
        st.bar_chart(model["waterfall_rows"], x="stage", y="elapsed_ms")
    st.dataframe(_stage_table_rows(model["stage_rows"]), use_container_width=True, hide_index=True)

    dense_tab, sparse_tab, fusion_tab, rerank_tab, details_tab = st.tabs(
        ["Dense", "Sparse", "Fusion", "Rerank", "Details"]
    )
    with dense_tab:
        _render_rows(st, model["dense_rows"], "No dense result details recorded.")
    with sparse_tab:
        _render_rows(st, model["sparse_rows"], "No sparse result details recorded.")
    with fusion_tab:
        _render_rows(st, model["fusion_rows"], "No fusion result details recorded.")
    with rerank_tab:
        _render_rows(st, model["rerank_rows"], "No rerank result details recorded.")
    with details_tab:
        for row in model["stage_rows"]:
            with st.expander(row["stage"], expanded=False):
                st.json(row["details"])


def _render_rows(st: Any, rows: list[dict[str, Any]], empty_message: str) -> None:
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info(empty_message)


def _filter_summaries(summaries: list[dict[str, Any]], search: str) -> list[dict[str, Any]]:
    term = search.strip().lower()
    if not term:
        return summaries
    return [
        summary
        for summary in summaries
        if term in " ".join(str(summary.get(key) or "").lower() for key in ("query", "trace_id", "collection"))
    ]


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


def _query_stats(trace: dict[str, Any] | None) -> dict[str, Any]:
    stats = {
        "dense_count": 0,
        "sparse_count": 0,
        "fused_count": 0,
        "result_count": 0,
        "rerank_enabled": False,
        "failed": False,
    }
    if trace is None:
        return stats
    for stage in _stages(trace):
        name = str(stage.get("name") or "")
        data = _data(stage)
        stats["failed"] = stats["failed"] or "failed" in name
        if "dense" in name:
            stats["dense_count"] = max(stats["dense_count"], _int_value(data.get("result_count"), len(_results_from_data(data))))
        if "sparse" in name:
            stats["sparse_count"] = max(stats["sparse_count"], _int_value(data.get("result_count"), len(_results_from_data(data))))
        if "fusion" in name or "hybrid_search.completed" == name:
            stats["fused_count"] = max(stats["fused_count"], _int_value(data.get("result_count"), len(_results_from_data(data))))
        if "rerank" in name or name == "query.completed":
            stats["rerank_enabled"] = stats["rerank_enabled"] or bool(data.get("rerank_enabled") is True)
            stats["result_count"] = max(stats["result_count"], _int_value(data.get("result_count"), len(_results_from_data(data))))
    return stats


def _route_rows(trace: dict[str, Any] | None, *, route: str) -> list[dict[str, Any]]:
    if trace is None:
        return []
    rows: list[dict[str, Any]] = []
    for stage in _stages(trace):
        name = str(stage.get("name") or "")
        if route not in name:
            continue
        data = _data(stage)
        for rank, result in enumerate(_results_from_data(data, route=route), start=1):
            rows.append(_result_row(result, rank=rank, route=route))
    return rows


def _fusion_rows(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    return _ranked_rows_from_trace(trace, name_contains=("fusion",), route="fusion")


def _rerank_rows(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if trace is None:
        return []
    fused = _fusion_rows(trace)
    fused_rank_by_id = {row["chunk_id"]: row["rank"] for row in fused}
    rows = _ranked_rows_from_trace(trace, name_contains=("rerank",), route="rerank")
    for row in rows:
        previous_rank = fused_rank_by_id.get(row["chunk_id"])
        row["previous_rank"] = previous_rank
        row["rank_delta"] = None if previous_rank is None else previous_rank - int(row["rank"])
    return rows


def _ranked_rows_from_trace(trace: dict[str, Any] | None, *, name_contains: tuple[str, ...], route: str) -> list[dict[str, Any]]:
    if trace is None:
        return []
    rows = []
    for stage in _stages(trace):
        name = str(stage.get("name") or "")
        if not any(token in name for token in name_contains):
            continue
        for rank, result in enumerate(_results_from_data(_data(stage), route=route), start=1):
            rows.append(_result_row(result, rank=rank, route=route))
    return rows


def _results_from_data(data: dict[str, Any], route: str | None = None) -> list[dict[str, Any]]:
    keys = [
        "results",
        "items",
        "candidates",
        "retrieval_results",
        "fusion_results",
        "fused_results",
        "rerank_results",
        "reranked_results",
    ]
    if route:
        keys = [f"{route}_results", f"{route}_items", *keys]
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _result_row(result: dict[str, Any], *, rank: int, route: str) -> dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    return {
        "rank": _int_value(result.get("rank"), rank),
        "route": route,
        "chunk_id": result.get("chunk_id") or result.get("id"),
        "score": result.get("score"),
        "source_path": metadata.get("source_path") or result.get("source_path"),
        "text": _summarize(str(result.get("text") or "")),
    }


def _summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "trace_id": summary["trace_id"],
            "query": summary.get("query"),
            "status": summary["status"],
            "collection": summary.get("collection"),
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


def _stages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    stages = trace.get("stages")
    return [stage for stage in stages if isinstance(stage, dict)] if isinstance(stages, list) else []


def _data(stage: dict[str, Any]) -> dict[str, Any]:
    data = stage.get("data")
    return data if isinstance(data, dict) else {}


def _int_value(*values: Any) -> int:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _summarize(text: str, limit: int = 140) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3]}..."


__all__ = ["query_traces_model", "render"]
