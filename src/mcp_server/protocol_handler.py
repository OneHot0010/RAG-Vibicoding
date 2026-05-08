"""Minimal JSON-RPC protocol handling for the MCP stdio server."""

from __future__ import annotations

from typing import Any


MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "rag-vibecoding"
SERVER_VERSION = "0.1.0"


def handle_jsonrpc_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message for the E1 server lifecycle."""
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        return _success_response(request_id, _initialize_result(message.get("params")))
    if method == "notifications/initialized":
        return None
    return _error_response(request_id, -32601, f"Method not found: {method}")


def _initialize_result(params: Any) -> dict[str, Any]:
    client_protocol = params.get("protocolVersion") if isinstance(params, dict) else None
    return {
        "protocolVersion": client_protocol or MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def _success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
