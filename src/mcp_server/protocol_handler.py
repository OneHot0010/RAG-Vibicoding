"""JSON-RPC protocol handling for the MCP stdio server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "rag-vibecoding"
SERVER_VERSION = "0.1.0"

INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ProtocolError(ValueError):
    """Raised for JSON-RPC protocol validation errors."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolSpec:
    """MCP tool registration metadata and callable."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def to_mcp_tool(self) -> dict[str, Any]:
        """Serialize this tool in the MCP tools/list shape."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class ProtocolHandler:
    """Handle MCP JSON-RPC requests and map errors to standard codes."""

    def __init__(self, tools: Mapping[str, ToolSpec] | None = None) -> None:
        self.tools = dict(tools or {})

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded JSON-RPC message."""
        request_id = message.get("id") if isinstance(message, dict) else None
        try:
            request = _validate_request(message)
            request_id = request.get("id")
            if request_id is None:
                return None
            method = request["method"]
            if method == "initialize":
                return _success_response(request_id, self.handle_initialize(request.get("params")))
            if method == "notifications/initialized":
                return None
            if method == "tools/list":
                return _success_response(request_id, self.handle_tools_list())
            if method == "tools/call":
                return _success_response(request_id, self.handle_tools_call(request.get("params")))
            raise ProtocolError(METHOD_NOT_FOUND, f"Method not found: {method}")
        except ProtocolError as exc:
            return _error_response(request_id, exc.code, exc.message)
        except Exception:
            return _error_response(request_id, INTERNAL_ERROR, "Internal error")

    def handle_initialize(self, params: Any) -> dict[str, Any]:
        """Return MCP initialize capabilities and server metadata."""
        if params is not None and not isinstance(params, dict):
            raise ProtocolError(INVALID_PARAMS, "initialize params must be an object")
        client_protocol = params.get("protocolVersion") if isinstance(params, dict) else None
        return {
            "protocolVersion": client_protocol or MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def handle_tools_list(self) -> dict[str, Any]:
        """Return registered tool schemas."""
        return {"tools": [tool.to_mcp_tool() for tool in self.tools.values()]}

    def handle_tools_call(self, params: Any) -> dict[str, Any]:
        """Route a tools/call request to a registered tool."""
        if not isinstance(params, dict):
            raise ProtocolError(INVALID_PARAMS, "tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise ProtocolError(INVALID_PARAMS, "tools/call params.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ProtocolError(INVALID_PARAMS, "tools/call params.arguments must be an object")
        tool = self.tools.get(name)
        if tool is None:
            raise ProtocolError(INVALID_PARAMS, f"Unknown tool: {name}")
        try:
            result = tool.handler(arguments)
        except ProtocolError:
            raise
        except Exception:
            raise ProtocolError(INTERNAL_ERROR, "Internal error") from None
        if not isinstance(result, dict):
            raise ProtocolError(INTERNAL_ERROR, "Internal error")
        return result


def default_protocol_handler() -> ProtocolHandler:
    """Return the default protocol handler used by the stdio server."""
    from mcp_server.tools import query_knowledge_hub_tool_spec

    tool = query_knowledge_hub_tool_spec()
    return ProtocolHandler({tool.name: tool})


def handle_jsonrpc_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Backward-compatible helper for server transport tests."""
    return default_protocol_handler().handle_message(message)


def _validate_request(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError(INVALID_REQUEST, "Invalid Request")
    if message.get("jsonrpc") != "2.0":
        raise ProtocolError(INVALID_REQUEST, "Invalid Request")
    method = message.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError(INVALID_REQUEST, "Invalid Request")
    return message


def _success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
