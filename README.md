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
- `C6` metadata enrichment: rule title/summary/tags, optional structured LLM metadata, fallback handling, trace hooks, and serialization tests.
- `C7` image captioning: optional Vision LLM captions for chunk image refs, metadata writeback, disabled/error fallback, prompt templating, and trace tests.
- `C8` dense encoding: `DenseEncoder`, embedding factory integration, `ChunkRecord` dense vectors, dimension validation, trace hooks, and mock embedding tests.
- `C9` sparse encoding: BM25-ready term-frequency `ChunkRecord`s, document frequency stats, average doc length, token filtering, trace hooks, and edge-case tests.
- `C10` batch processing: stable chunk batching, dense/sparse encoding orchestration, flattened records, per-batch timing, trace events, and serialization tests.
- `C11` BM25 indexing: inverted index build/add, IDF calculation, stable query ranking, JSON persistence/load, trace hooks, and roundtrip tests.
- `C12` vector upsert: deterministic content-based vector ids, VectorStore upsert adapter, idempotent writes, batch ordering, trace hooks, and mock store tests.
- `C13` image storage: SQLite image index, WAL mode, image file persistence, collection/doc_hash listing, deletion, trace hooks, and concurrency tests.
- `C14` ingestion pipeline: integrity/load/split/transform/encode/store orchestration, skip/force handling, BM25/Chroma/ImageStorage outputs, and failure-state tests.
- `C15` offline ingestion CLI: `scripts/ingest.py` with `--path`, `--collection`, `--force`, local deterministic embeddings, data-dir routing, JSON summaries, and e2e tests.
- `D1` query processing: `ProcessedQuery`, keyword extraction, inline/explicit metadata filters, value coercion, trace hooks, and preprocessing tests.
- `D2` dense retrieval: `RetrievalResult`, query embedding orchestration, VectorStore querying, metadata filters, trace hooks, and mock retriever tests.
- `D3` sparse retrieval: VectorStore `get_by_ids`, Chroma record hydration, BM25 keyword retrieval, original chunk id mapping, trace hooks, and sparse retriever tests.
- `D4` RRF fusion: dense/sparse rank fusion, configurable k, deterministic tie-breaking, contribution metadata, trace hooks, and fusion tests.
- `D5` hybrid search: QueryProcessor/Dense/Sparse/RRF orchestration, metadata filter fallback, single-route degradation, trace events, and integration tests.
- `D6` core reranker: RetrievalResult-to-reranker adapter, backend rerank integration, fusion-order fallback metadata, trace hooks, and fallback tests.
- `D7` query CLI: `scripts/query.py` with query/top-k/collection/verbose/no-rerank options, offline embeddings, formatted results, no-data messaging, and e2e tests.
- `E1` MCP stdio server: line-delimited JSON-RPC entry point, initialize response, stderr-only logging, clean stdout transport, and subprocess integration tests.
- `E2` protocol handler: JSON-RPC request validation, initialize/tools/list/tools/call routing, tool schemas, standard error codes, and handler tests.
- `E3` query knowledge hub tool: MCP tool registration, HybridSearch/CoreReranker execution, Markdown answers, structured citations, no-result response, and MCP integration tests.
- `E4` list collections tool: MCP tool registration, document directory scanning, Chroma/Image index statistics, structured collection metadata, and stdio integration tests.
- `E5` get document summary tool: MCP tool registration, Chroma metadata lookup by doc id/source name/hash, title/summary/tag aggregation, ingestion timestamp lookup, and stdio integration tests.
- `E6` multimodal response assembly: image ref resolution from chunk metadata/ImageStorage, base64 MCP ImageContent, mime type detection, skipped-image metadata, and ResponseBuilder integration tests.
- `F1` trace context enhancement: query/ingestion trace types, finish/elapsed timing, JSON-safe serialization, compatibility-preserving stage records, and TraceCollector tests.
- `F2` JSON Lines trace logger: JSONFormatter, trace file handler, write_trace persistence, duplicate-handler protection, and TraceCollector sink integration tests.
- `F3` query trace integration: CLI/MCP query entrypoints create query traces, record query lifecycle stages, persist JSONL traces, and expose MCP structured trace metadata.
- `F4` ingestion trace integration: ingest CLI creates ingestion traces per file, records pipeline lifecycle stages, persists JSONL traces, and keeps pipeline default traces typed as ingestion.
- `F5` pipeline progress callbacks: optional `on_progress(stage, current, total)` support for ingestion stages, skip/failure completion events, and progress callback tests.
- `G1` dashboard foundation: Streamlit entrypoint, six-page navigation shell, overview model/page, configuration cards, data asset statistics, and start script.

## Quick Check

```powershell
python -m compileall src
python -c "import mcp_server; import core; import ingestion; import libs; import observability"
.\.venv\Scripts\Activate.ps1; pytest -q
.\.venv\Scripts\Activate.ps1; python main.py
```
