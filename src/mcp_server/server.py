"""MCP server stdio transport entry point."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from mcp_server.protocol_handler import handle_jsonrpc_message
from observability.logger import get_logger


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    """Run a line-delimited JSON-RPC stdio server."""
    logger = get_logger("mcp_server")
    logger.info("MCP stdio server starting")
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                response = _error_response(None, -32600, "Invalid Request")
            else:
                response = handle_jsonrpc_message(message)
        except json.JSONDecodeError:
            response = _error_response(None, -32700, "Parse error")
        except Exception as exc:
            print(f"ERROR:mcp_server:{exc}", file=stderr)
            response = _error_response(None, -32603, "Internal error")

        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            stdout.flush()
    logger.info("MCP stdio server stopped")
    return 0


def main() -> int:
    """Console entry point."""
    return serve()


def _error_response(request_id: object, code: int, message: str) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
