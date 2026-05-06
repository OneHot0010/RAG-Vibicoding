# RAG Vibecoding

Spec-driven RAG and MCP knowledge hub. The current milestone contains the initial project scaffold; feature modules are implemented incrementally from `DEV_SPEC.md`.

## Current Status

- `A1` scaffold: source tree, configuration placeholders, prompt placeholders, data/cache/log directories, and minimal entry point.
- `A2` testing baseline: pytest configuration, test directory conventions, smoke import tests, and a minimal sample document fixture.

## Quick Check

```powershell
python -m compileall src
python -c "import mcp_server; import core; import ingestion; import libs; import observability"
.\.venv\Scripts\Activate.ps1; pytest -q
```
