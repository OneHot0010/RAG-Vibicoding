"""E2E tests for the developer query CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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


def test_query_cli_returns_results_after_ingestion(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "azure.pdf"
    data_dir = tmp_path / "data"
    write_minimal_pdf(pdf_path, "Azure configuration uses endpoints deployments and api versions")

    ingest = run_command(
        repo_root,
        [
            "scripts/ingest.py",
            "--path",
            str(pdf_path),
            "--collection",
            "docs",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert ingest.returncode == 0, ingest.stderr

    query = run_command(
        repo_root,
        [
            "scripts/query.py",
            "--query",
            "Azure endpoints",
            "--collection",
            "docs",
            "--top-k",
            "3",
            "--data-dir",
            str(data_dir),
            "--no-rerank",
        ],
    )

    assert query.returncode == 0, query.stderr
    assert "Top-K 检索结果" in query.stdout
    assert "azure.pdf" in query.stdout
    assert "Azure configuration" in query.stdout


def test_query_cli_writes_query_trace_jsonl(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "trace.pdf"
    data_dir = tmp_path / "data"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    write_minimal_pdf(pdf_path, "Trace logging records query retrieval stages")
    ingest = run_command(
        repo_root,
        ["scripts/ingest.py", "--path", str(pdf_path), "--collection", "docs", "--data-dir", str(data_dir)],
    )
    assert ingest.returncode == 0, ingest.stderr

    query = run_command(
        repo_root,
        [
            "scripts/query.py",
            "--query",
            "trace logging",
            "--collection",
            "docs",
            "--data-dir",
            str(data_dir),
            "--trace-log-file",
            str(trace_log),
            "--no-rerank",
        ],
    )

    assert query.returncode == 0, query.stderr
    rows = [json.loads(line) for line in trace_log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trace_type"] == "query"
    assert rows[0]["finished_at"] is not None
    stage_names = [stage["name"] for stage in rows[0]["stages"]]
    assert "query.start" in stage_names
    assert "hybrid_search.completed" in stage_names
    assert "query.completed" in stage_names


def test_query_cli_verbose_outputs_trace_and_rerank_sections(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "rag.pdf"
    data_dir = tmp_path / "data"
    write_minimal_pdf(pdf_path, "RAG retrieval combines dense sparse fusion and reranking")

    ingest = run_command(
        repo_root,
        ["scripts/ingest.py", "--path", str(pdf_path), "--collection", "docs", "--data-dir", str(data_dir)],
    )
    assert ingest.returncode == 0, ingest.stderr

    query = run_command(
        repo_root,
        [
            "scripts/query.py",
            "--query",
            "dense sparse fusion",
            "--collection",
            "docs",
            "--data-dir",
            str(data_dir),
            "--verbose",
        ],
    )

    assert query.returncode == 0, query.stderr
    assert "Verbose" in query.stdout
    assert "fusion_results" in query.stdout
    assert "rerank_results" in query.stdout
    assert "hybrid_search.completed" in query.stdout
    assert '"trace_type": "query"' in query.stdout


def test_query_cli_without_indexes_has_friendly_message(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    query = run_command(
        repo_root,
        ["scripts/query.py", "--query", "anything", "--data-dir", str(tmp_path / "missing-data")],
    )

    assert query.returncode == 0
    assert "未找到相关文档" in query.stdout
    assert query.stderr == ""
