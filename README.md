# RAG Vibecoding

Spec-driven RAG and MCP knowledge hub. The current milestone has completed the Libs pluggable layer; feature modules are implemented incrementally from `DEV_SPEC.md`.

## Current Status

- `A1` scaffold: source tree, configuration placeholders, prompt placeholders, data/cache/log directories, and minimal entry point.
- `A2` testing baseline: pytest configuration, test directory conventions, smoke import tests, and a minimal sample document fixture.
- `A3` settings loader: YAML-backed dataclass settings, fail-fast validation, stderr logger, and startup config check.
- `B1` LLM abstraction: `BaseLLM`, normalized chat messages, registry-backed `LLMFactory`, and fake-provider routing tests.
- `B2` Embedding abstraction: `BaseEmbedding`, registry-backed `EmbeddingFactory`, and stable fake-vector routing tests.
- `B3` Splitter abstraction: `BaseSplitter`, splitter settings, registry-backed `SplitterFactory`, and fake-strategy routing tests.
- `B4` VectorStore abstraction: `VectorRecord`, `VectorQueryResult`, `BaseVectorStore`, registry-backed `VectorStoreFactory`, and contract tests.
- `B5` Reranker abstraction: `RerankCandidate`, `RerankResult`, `BaseReranker`, `NoneReranker`, and registry-backed `RerankerFactory`.
- `B6` Evaluator abstraction: `EvaluationCase`, `EvaluationResult`, `BaseEvaluator`, `EvaluatorFactory`, and deterministic `CustomEvaluator` metrics.
- `B7.1` OpenAI-compatible LLMs: OpenAI, Azure OpenAI, and DeepSeek chat providers with mocked HTTP smoke tests.
- `B7.2` Ollama LLM: local `/api/chat` provider, factory registration, mocked HTTP tests, and readable connection errors.
- `B7.3` OpenAI/Azure embeddings: batch embedding providers, Azure deployment URL handling, factory registration, and mocked HTTP tests.
- `B7.4` Ollama embeddings: local `/api/embeddings` provider, batch wrapper, factory registration, and mocked HTTP/error tests.
- `B7.5` Recursive splitter: Markdown-aware default splitter, fenced code preservation, overlap support, and factory registration.
- `B7.6` ChromaStore default backend: local persistent vector store, cosine query, top-k, metadata filters, and roundtrip integration tests.
- `B7.7` LLM reranker: prompt-backed ranked-id reranking, schema validation, omitted-candidate fallback, and mock LLM tests.
- `B7.8` Cross-Encoder reranker: injectable scorer, top-M reranking, deterministic keyword fallback, factory registration, and mock scorer tests.
- `B8` Vision LLM abstraction: `BaseVisionLLM`, normalized `ChatResponse`, independent vision provider registry, and fake-provider factory tests.
- `B9` Azure Vision LLM: Azure multimodal chat payloads, file/bytes image encoding, deployment URLs, factory registration, and mocked HTTP tests.
- `C1` core data contracts: `Document`, `Chunk`, `ChunkRecord`, `ImageRef`, image placeholder helpers, metadata validation, and JSON-stable serialization tests.
- `C2` file integrity: SHA256 calculation, SQLite ingestion history, WAL mode, success/failed state tracking, and concurrent write tests.
- `C3` PDF loading: `BaseLoader`, dependency-light `PdfLoader`, text extraction fallback, image placeholder metadata, and degradation tests.
- `C4` document chunking: `DocumentChunker`, splitter factory integration, stable chunk ids, source offsets, source refs, and chunk-local image metadata.
- `C5` chunk refinement: transform base contract, rule-based cleanup, optional LLM rewrite, fallback metadata, trace hooks, noisy fixtures, and opt-in real LLM test.

## Quick Check

```powershell
python -m compileall src
python -c "import mcp_server; import core; import ingestion; import libs; import observability"
.\.venv\Scripts\Activate.ps1; pytest -q
.\.venv\Scripts\Activate.ps1; python main.py
```
