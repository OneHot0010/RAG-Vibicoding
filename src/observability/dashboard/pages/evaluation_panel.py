"""Dashboard evaluation panel page."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services.evaluation_service import EvaluationService


def evaluation_panel_model(
    evaluation_service: EvaluationService | None = None,
    *,
    test_set_path: str | None = None,
    last_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return display data for the evaluation panel."""
    service = evaluation_service or EvaluationService()
    summary_error = None
    try:
        test_set = service.test_set_summary(test_set_path)
    except Exception as exc:
        test_set = {"path": test_set_path or service.default_test_set_path(), "case_count": 0, "queries": []}
        summary_error = str(exc)
    return {
        "backends": service.available_backends(),
        "test_set": test_set,
        "summary_error": summary_error,
        "report": last_report,
        "metric_rows": _metric_rows(last_report),
        "case_rows": _case_rows(last_report),
    }


def render(
    st: Any | None = None,
    *,
    evaluation_service: EvaluationService | None = None,
) -> dict[str, Any]:
    """Render the evaluation panel and return the underlying model for tests."""
    service = evaluation_service or EvaluationService()
    model = evaluation_panel_model(service)
    if st is None:
        return model

    st.title("Evaluation")
    controls = st.columns([2, 1, 1])
    test_set_path = controls[0].text_input("Golden test set", value=model["test_set"]["path"])
    backends = model["backends"] or ["custom"]
    selected_backend = controls[1].selectbox("Backend", backends)
    top_k = controls[2].number_input("Top K", min_value=1, value=3, step=1)
    online_embedding = st.checkbox("Use online embedding provider", value=False)

    model = evaluation_panel_model(service, test_set_path=test_set_path)
    if model["summary_error"]:
        st.warning(model["summary_error"])
    else:
        st.caption(f"{model['test_set']['case_count']} golden cases")

    if st.button("Run evaluation"):
        try:
            report = service.run_evaluation(
                test_set_path=test_set_path,
                backend=selected_backend,
                top_k=int(top_k),
                online_embedding=online_embedding,
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
        else:
            st.success("Evaluation completed.")
            model = evaluation_panel_model(service, test_set_path=test_set_path, last_report=report)

    _render_report(st, model)
    return model


def _render_report(st: Any, model: dict[str, Any]) -> None:
    if model["report"] is None:
        st.info("Run evaluation to view metrics.")
        return
    st.subheader("Metrics")
    metric_columns = st.columns(max(1, min(4, len(model["metric_rows"]))))
    for index, row in enumerate(model["metric_rows"]):
        metric_columns[index % len(metric_columns)].metric(row["metric"], row["value"])
    st.dataframe(model["metric_rows"], use_container_width=True, hide_index=True)

    st.subheader("Cases")
    st.dataframe(model["case_rows"], use_container_width=True, hide_index=True)
    for case in model["report"].get("cases", []):
        with st.expander(str(case.get("query") or "")):
            st.write(
                {
                    "metrics": case.get("metrics"),
                    "retrieved_ids": case.get("retrieved_ids"),
                    "golden_ids": case.get("golden_ids"),
                }
            )
            st.json(case.get("details") or {})


def _metric_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return []
    return [{"metric": str(name), "value": round(float(value), 6)} for name, value in sorted(metrics.items())]


def _case_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    rows = []
    for case in report.get("cases", []):
        if not isinstance(case, dict):
            continue
        metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
        rows.append(
            {
                "query": case.get("query"),
                "hit_rate": metrics.get("hit_rate"),
                "mrr": metrics.get("mrr"),
                "retrieved": len(case.get("retrieved_ids") or []),
                "golden": len(case.get("golden_ids") or []),
            }
        )
    return rows


__all__ = ["evaluation_panel_model", "render"]
