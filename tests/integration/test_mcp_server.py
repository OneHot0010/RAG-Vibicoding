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
