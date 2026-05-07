# RAG Vibecoding

Spec-driven RAG and MCP knowledge hub. The current milestone contains the initial project scaffold; feature modules are implemented incrementally from `DEV_SPEC.md`.

## Current Status

- `A1` scaffold: source tree, configuration placeholders, prompt placeholders, data/cache/log directories, and minimal entry point.
- `A2` testing baseline: pytest configuration, test directory conventions, smoke import tests, and a minimal sample document fixture.
- `A3` settings loader: YAML-backed dataclass settings, fail-fast validation, stderr logger, and startup config check.
- `B1` LLM abstraction: `BaseLLM`, normalized chat messages, registry-backed `LLMFactory`, and fake-provider routing tests.
- `B2` Embedding abstraction: `BaseEmbedding`, registry-backed `EmbeddingFactory`, and stable fake-vector routing tests.
- `B3` Splitter abstraction: `BaseSplitter`, splitter settings, registry-backed `SplitterFactory`, and fake-strategy routing tests.
- `B4` VectorStore abstraction: `VectorRecord`, `VectorQueryResult`, `BaseVectorStore`, registry-backed `VectorStoreFactory`, and contract tests.
- `B5` Reranker abstraction: `RerankCandidate`, `RerankResult`, `BaseReranker`, `NoneReranker`, and registry-backed `RerankerFactory`.

## Quick Check

```powershell
python -m compileall src
python -c "import mcp_server; import core; import ingestion; import libs; import observability"
.\.venv\Scripts\Activate.ps1; pytest -q
.\.venv\Scripts\Activate.ps1; python main.py
```
