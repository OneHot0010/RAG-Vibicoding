"""E2E recall regression test using the golden evaluation runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from observability.dashboard.pages import data_browser, evaluation_panel, ingestion_traces, query_traces
from observability.dashboard.services.data_service import DataService
from observability.dashboard.services.evaluation_service import EvaluationService


MIN_HIT_RATE = 1.0


def write_minimal_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
    path.write_text(
        "\n".join(
            [
                "%PDF-1.4",
                "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
                "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
                "3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj",
                f"4 0 obj << /Length {len(escaped) + 40} >>",
                "stream",
                "BT",
                "/F1 12 Tf",
                f"72 720 Td ({escaped}) Tj",
                "ET",
                "endstream",
                "endobj",
                "%%EOF",
            ]
        ),
        encoding="latin-1",
    )


def run_command(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def run_mcp_server(repo_root: Path, request: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def write_golden(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "azure endpoint deployment api key",
                        "expected_sources": ["azure_recall.pdf"],
                        "filters": {"collection": "recall"},
                        "top_k": 3,
                    },
                    {
                        "query": "hybrid dense sparse fusion retrieval",
                        "expected_sources": ["hybrid_recall.pdf"],
                        "filters": {"collection": "recall"},
                        "top_k": 3,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_recall_regression_meets_hit_rate_threshold(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = tmp_path / "docs"
    data_dir = tmp_path / "data"
    golden = tmp_path / "golden_recall.json"
    docs_dir.mkdir()
    write_minimal_pdf(
        docs_dir / "azure_recall.pdf",
        "Azure endpoint deployment api key configuration guide for recall regression",
    )
    write_minimal_pdf(
        docs_dir / "hybrid_recall.pdf",
        "Hybrid retrieval combines dense sparse fusion ranking for recall regression",
    )
    write_golden(golden)

    ingest = run_command(
        repo_root,
        [
            "scripts/ingest.py",
            "--path",
            str(docs_dir),
            "--collection",
            "recall",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert ingest.returncode == 0, ingest.stderr

    evaluation = run_command(
        repo_root,
        [
            "scripts/evaluate.py",
            "--data-dir",
            str(data_dir),
            "--test-set",
            str(golden),
            "--backend",
            "custom",
            "--top-k",
            "3",
        ],
    )

    assert evaluation.returncode == 0, evaluation.stderr
    payload = json.loads(evaluation.stdout)
    assert payload["case_count"] == 2
    assert payload["metrics"]["hit_rate"] >= MIN_HIT_RATE
    assert all(case["metrics"]["hit_rate"] == 1.0 for case in payload["cases"])


def test_full_chain_acceptance_ingest_mcp_dashboard_and_evaluate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = tmp_path / "docs"
    data_dir = tmp_path / "data"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    golden = tmp_path / "golden_acceptance.json"
    pdf_path = docs_dir / "acceptance_full_chain.pdf"
    docs_dir.mkdir()
    write_minimal_pdf(
        pdf_path,
        "Acceptance full chain query validates ingest MCP dashboard traces and evaluation",
    )
    golden.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "acceptance full chain dashboard evaluation",
                        "expected_sources": [pdf_path.name],
                        "filters": {"collection": "acceptance"},
                        "top_k": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    ingest = run_command(
        repo_root,
        [
            "scripts/ingest.py",
            "--path",
            str(docs_dir),
            "--collection",
            "acceptance",
            "--data-dir",
            str(data_dir),
            "--trace-log-file",
            str(trace_log),
        ],
    )
    assert ingest.returncode == 0, ingest.stderr
    ingest_payload = json.loads(ingest.stdout)
    assert ingest_payload["ingested"] == 1

    cli_query = run_command(
        repo_root,
        [
            "scripts/query.py",
            "--query",
            "acceptance full chain",
            "--collection",
            "acceptance",
            "--top-k",
            "3",
            "--data-dir",
            str(data_dir),
            "--trace-log-file",
            str(trace_log),
            "--verbose",
            "--no-rerank",
        ],
    )
    assert cli_query.returncode == 0, cli_query.stderr
    assert "Top-K 检索结果" in cli_query.stdout
    assert pdf_path.name in cli_query.stdout

    mcp_query = run_mcp_server(
        repo_root,
        {
            "jsonrpc": "2.0",
            "id": "acceptance-query",
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "dashboard evaluation citations",
                    "collection": "acceptance",
                    "top_k": 3,
                    "data_dir": str(data_dir),
                    "trace_log_file": str(trace_log),
                    "no_rerank": True,
                },
            },
        },
    )
    assert mcp_query.returncode == 0, mcp_query.stderr
    mcp_payload = json.loads(mcp_query.stdout)
    citations = mcp_payload["result"]["structuredContent"]["citations"]
    assert citations
    assert citations[0]["source"].endswith(pdf_path.name)

    data_model = data_browser.data_browser_model(data_service=DataService(data_dir=data_dir))
    ingestion_model = ingestion_traces.ingestion_traces_model(trace_log_file=trace_log)
    query_model = query_traces.query_traces_model(trace_log_file=str(trace_log))
    evaluation_model = evaluation_panel.evaluation_panel_model(
        EvaluationService(data_dir=data_dir),
        test_set_path=str(golden),
    )
    assert data_model["documents"][0]["source_path"].endswith(pdf_path.name)
    assert ingestion_model["selected_summary"]["trace_type"] == "ingestion"
    assert query_model["selected_summary"]["trace_type"] == "query"
    assert evaluation_model["test_set"]["case_count"] == 1

    evaluation = run_command(
        repo_root,
        [
            "scripts/evaluate.py",
            "--data-dir",
            str(data_dir),
            "--test-set",
            str(golden),
            "--backend",
            "custom",
            "--top-k",
            "3",
        ],
    )
    assert evaluation.returncode == 0, evaluation.stderr
    eval_payload = json.loads(evaluation.stdout)
    assert eval_payload["case_count"] == 1
    assert eval_payload["metrics"]["hit_rate"] == 1.0
