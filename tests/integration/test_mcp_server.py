"""Integration tests for the MCP stdio server entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_server(repo_root: Path, payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=payload,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


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


def run_ingest(repo_root: Path, pdf_path: Path, data_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/ingest.py",
            "--path",
            str(pdf_path),
            "--collection",
            "docs",
            "--data-dir",
            str(data_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


def test_mcp_server_initialize_over_stdio() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "pytest", "version": "0"}},
    }

    completed = run_server(repo_root, json.dumps(request) + "\n")

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == "2025-06-18"
    assert response["result"]["capabilities"] == {"tools": {}}
    assert response["result"]["serverInfo"]["name"] == "rag-vibecoding"
    assert "MCP stdio server starting" in completed.stderr
    assert "MCP stdio server stopped" in completed.stderr


def test_mcp_server_stdout_is_not_polluted_by_logs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    request = {"jsonrpc": "2.0", "id": "abc", "method": "unknown/method"}

    completed = run_server(repo_root, json.dumps(request) + "\n")

    assert completed.returncode == 0
    response = json.loads(completed.stdout)
    assert response == {
        "jsonrpc": "2.0",
        "id": "abc",
        "error": {"code": -32601, "message": "Method not found: unknown/method"},
    }
    assert completed.stdout.strip().startswith("{")
    assert "INFO:mcp_server" not in completed.stdout
    assert "INFO:mcp_server" in completed.stderr


def test_mcp_server_parse_error_is_jsonrpc_response() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    completed = run_server(repo_root, "{not json}\n")

    assert completed.returncode == 0
    assert json.loads(completed.stdout) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }


def test_query_knowledge_hub_tool_call_returns_markdown_and_citations(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "azure.pdf"
    data_dir = tmp_path / "data"
    write_minimal_pdf(pdf_path, "Azure configuration uses endpoints deployments and api versions")
    ingest = run_ingest(repo_root, pdf_path, data_dir)
    assert ingest.returncode == 0, ingest.stderr
    request = {
        "jsonrpc": "2.0",
        "id": "query",
        "method": "tools/call",
        "params": {
            "name": "query_knowledge_hub",
            "arguments": {
                "query": "Azure endpoints",
                "collection": "docs",
                "top_k": 3,
                "data_dir": str(data_dir),
                "no_rerank": True,
            },
        },
    }

    completed = run_server(repo_root, json.dumps(request) + "\n")

    assert completed.returncode == 0
    response = json.loads(completed.stdout)
    result = response["result"]
    assert result["content"][0]["type"] == "text"
    assert "[1]" in result["content"][0]["text"]
    assert "Azure configuration" in result["content"][0]["text"]
    assert result["structuredContent"]["citations"][0]["source"].endswith("azure.pdf")
    assert result["structuredContent"]["citations"][0]["chunk_id"]
    assert result["structuredContent"]["citations"][0]["score"] > 0


def test_list_collections_tool_call_returns_structured_collection_stats(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs = tmp_path / "data" / "documents" / "docs"
    docs.mkdir(parents=True)
    (docs / "guide.md").write_text("guide", encoding="utf-8")
    request = {
        "jsonrpc": "2.0",
        "id": "collections",
        "method": "tools/call",
        "params": {
            "name": "list_collections",
            "arguments": {"data_dir": str(tmp_path / "data")},
        },
    }

    completed = run_server(repo_root, json.dumps(request) + "\n")

    assert completed.returncode == 0
    response = json.loads(completed.stdout)
    result = response["result"]
    assert result["content"][0]["type"] == "text"
    assert "docs" in result["content"][0]["text"]
    assert result["structuredContent"]["collections"][0]["name"] == "docs"
    assert result["structuredContent"]["collections"][0]["document_count"] == 1
