"""Tests for MCP JSON-RPC protocol handling."""

from __future__ import annotations

from mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    ProtocolHandler,
    ToolSpec,
    handle_jsonrpc_message,
)


def tool_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo input text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda arguments: {
            "content": [{"type": "text", "text": arguments["text"]}],
            "structuredContent": {"echo": arguments["text"]},
        },
    )


def test_initialize_returns_capabilities_and_server_info() -> None:
    response = ProtocolHandler().handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "rag-vibecoding", "version": "0.1.0"},
        },
    }


def test_notifications_do_not_return_response() -> None:
    assert ProtocolHandler().handle_message(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}
    ) is None


def test_tools_list_returns_registered_schema() -> None:
    response = ProtocolHandler({"echo": tool_spec()}).handle_message(
        {"jsonrpc": "2.0", "id": "list", "method": "tools/list"}
    )

    assert response["result"] == {
        "tools": [
            {
                "name": "echo",
                "description": "Echo input text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
        ]
    }


def test_tools_call_routes_to_registered_handler() -> None:
    response = ProtocolHandler({"echo": tool_spec()}).handle_message(
        {
            "jsonrpc": "2.0",
            "id": "call",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello"}},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": "call",
        "result": {
            "content": [{"type": "text", "text": "hello"}],
            "structuredContent": {"echo": "hello"},
        },
    }


def test_invalid_request_returns_32600() -> None:
    response = ProtocolHandler().handle_message({"id": 1, "method": "initialize"})

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": INVALID_REQUEST, "message": "Invalid Request"},
    }


def test_unknown_method_returns_32601() -> None:
    response = ProtocolHandler().handle_message({"jsonrpc": "2.0", "id": 1, "method": "missing"})

    assert response["error"] == {"code": METHOD_NOT_FOUND, "message": "Method not found: missing"}


def test_invalid_params_return_32602() -> None:
    handler = ProtocolHandler({"echo": tool_spec()})

    assert handler.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": []}
    )["error"] == {"code": INVALID_PARAMS, "message": "initialize params must be an object"}
    assert handler.handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": []}
    )["error"] == {"code": INVALID_PARAMS, "message": "tools/call params must be an object"}
    assert handler.handle_message(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "", "arguments": {}}}
    )["error"] == {
        "code": INVALID_PARAMS,
        "message": "tools/call params.name must be a non-empty string",
    }
    assert handler.handle_message(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "missing", "arguments": {}}}
    )["error"] == {"code": INVALID_PARAMS, "message": "Unknown tool: missing"}


def test_internal_tool_error_returns_32603_without_leaking_details() -> None:
    handler = ProtocolHandler(
        {
            "boom": ToolSpec(
                name="boom",
                description="Explodes.",
                input_schema={"type": "object"},
                handler=lambda arguments: (_ for _ in ()).throw(RuntimeError("secret stack detail")),
            )
        }
    )

    response = handler.handle_message(
        {"jsonrpc": "2.0", "id": "boom", "method": "tools/call", "params": {"name": "boom", "arguments": {}}}
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": "boom",
        "error": {"code": INTERNAL_ERROR, "message": "Internal error"},
    }


def test_handler_result_shape_error_is_internal_error() -> None:
    handler = ProtocolHandler(
        {
            "bad": ToolSpec(
                name="bad",
                description="Bad return.",
                input_schema={"type": "object"},
                handler=lambda arguments: ["not", "dict"],  # type: ignore[return-value]
            )
        }
    )

    response = handler.handle_message(
        {"jsonrpc": "2.0", "id": "bad", "method": "tools/call", "params": {"name": "bad", "arguments": {}}}
    )

    assert response["error"] == {"code": INTERNAL_ERROR, "message": "Internal error"}


def test_backward_compatible_helper_uses_default_handler() -> None:
    response = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert [tool["name"] for tool in response["result"]["tools"]] == ["query_knowledge_hub"]
