"""End-to-end MCP client simulation over the stdio transport."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


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


class StdioMCPClient:
    """Tiny JSON-RPC client used to exercise a long-lived server process."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def request(self, message: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        self._write(message)
        try:
            line = self._stdout_lines.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f"timed out waiting for response to {message['method']}") from None
        if line is None:
            raise AssertionError(f"server stdout closed before response to {message['method']}")
        return json.loads(line)

    def notify(self, message: dict[str, Any]) -> None:
        self._write(message)

    def close(self, timeout: float = 10.0) -> str:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)
            raise
        self._reader.join(timeout=timeout)
        return self.process.stderr.read() if self.process.stderr is not None else ""

    def _write(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            stripped = line.strip()
            if stripped:
                self._stdout_lines.put(stripped)
        self._stdout_lines.put(None)


def start_server(repo_root: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=repo_root,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_mcp_client_can_list_tools_and_query_knowledge_hub(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "mcp_client_guide.pdf"
    data_dir = tmp_path / "data"
    trace_log = tmp_path / "logs" / "query_traces.jsonl"
    write_minimal_pdf(pdf_path, "MCP client queries return citations from the knowledge hub")
    ingest = run_ingest(repo_root, pdf_path, data_dir)
    assert ingest.returncode == 0, ingest.stderr

    process = start_server(repo_root)
    client = StdioMCPClient(process)
    stderr = ""
    try:
        initialize = client.request(
            {
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "pytest-mcp-client", "version": "0"},
                },
            }
        )
        assert initialize["result"]["capabilities"] == {"tools": {}}

        client.notify({"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools_response = client.request({"jsonrpc": "2.0", "id": "tools", "method": "tools/list"})
        tool_names = [tool["name"] for tool in tools_response["result"]["tools"]]
        assert "query_knowledge_hub" in tool_names
        assert "list_collections" in tool_names

        query_response = client.request(
            {
                "jsonrpc": "2.0",
                "id": "query",
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        "query": "MCP client citations",
                        "collection": "docs",
                        "top_k": 3,
                        "data_dir": str(data_dir),
                        "no_rerank": True,
                        "trace_log_file": str(trace_log),
                    },
                },
            }
        )
    finally:
        stderr = client.close()

    assert process.returncode == 0
    assert "INFO:mcp_server" in stderr
    assert "INFO:mcp_server" not in json.dumps(query_response)
    assert query_response["jsonrpc"] == "2.0"
    assert query_response["id"] == "query"

    result = query_response["result"]
    assert result["content"][0]["type"] == "text"
    assert "[1]" in result["content"][0]["text"]
    assert "MCP client queries" in result["content"][0]["text"]
    citations = result["structuredContent"]["citations"]
    assert citations
    assert citations[0]["source"].endswith("mcp_client_guide.pdf")
    assert citations[0]["chunk_id"]
    assert result["structuredContent"]["trace"]["trace_type"] == "query"
    assert trace_log.is_file()
