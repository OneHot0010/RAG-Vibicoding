# RAG Vibecoding

Spec-driven local RAG and MCP knowledge hub. It ingests local PDF files, builds dense and BM25 indexes, exposes a query CLI, serves MCP tools over stdio, and provides a Streamlit Dashboard for browsing documents, traces, and evaluation results.

Current implementation status: **65/68 tasks complete (96%)**. The remaining stage focuses on README polish, contract cleanup, and final end-to-end acceptance.

## Quick Start

The default CLI path is offline-friendly: ingestion and query use deterministic local hash embeddings unless you opt into online providers with `--online-embedding`.

```powershell
cd D:\rag-vibecoding\RAG-Vibicoding
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Create or place one or more PDFs, then build the local indexes:

```powershell
python scripts\ingest.py --path tests\fixtures\sample_documents --collection docs --force
```

Query the knowledge hub:

```powershell
python scripts\query.py --query "What is in the sample document?" --collection docs --top-k 3 --no-rerank --verbose
```

Run evaluation against the bundled golden set:

```powershell
python scripts\evaluate.py --backend custom --top-k 3
```

Start the Dashboard:

```powershell
python -m pip install streamlit
python scripts\start_dashboard.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## Configuration

The main config file is [config/settings.yaml](config/settings.yaml). `${VAR}` and `%VAR%` environment variables are expanded when settings are loaded.

Important sections:

| Section | Purpose |
| --- | --- |
| `llm` | Chat model provider for LLM-backed features such as LLM reranking. Supported providers include `openai`, `azure`, `deepseek`, and `ollama`. |
| `embedding` | Online embedding provider when `--online-embedding` is used. The offline CLI path uses local hash embeddings instead. |
| `vision_llm` | Vision-capable model used by optional image captioning flows. |
| `splitter` | Chunking strategy and `chunk_size` / `chunk_overlap`. |
| `vector_store` | Vector backend and persistence path. The default backend is local JSON-backed `chroma` under `data/db/chroma`. |
| `retrieval` | Dense/sparse top-k settings and RRF fusion behavior. |
| `rerank` | Reranker backend. Use `none` for the most dependency-light path. |
| `evaluation` | Evaluation backends and golden test set path. |
| `observability` | JSON Lines trace logging, usually `logs/traces.jsonl`. |
| `dashboard` | Streamlit Dashboard settings such as port and refresh interval. |

Example online OpenAI-style settings:

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}

embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
```

Example Azure OpenAI settings:

```yaml
llm:
  provider: azure
  model: gpt-4o
  deployment: gpt-4o
  azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
  api_version: "2024-02-15-preview"
```

Example Ollama settings:

```yaml
llm:
  provider: ollama
  model: llama3.1
  base_url: http://localhost:11434
```

## CLI Usage

Ingest PDFs:

```powershell
python scripts\ingest.py --path .\data\documents --collection docs
python scripts\ingest.py --path .\data\documents\guide.pdf --collection docs --force
```

Useful ingestion flags:

| Flag | Meaning |
| --- | --- |
| `--path` | PDF file or directory containing PDFs. |
| `--collection` | Metadata collection name used for filtering later. |
| `--force` | Re-ingest even if the file hash already exists. |
| `--data-dir` | Alternate root for generated indexes and assets. |
| `--trace-log-file` | Override JSONL trace path. |
| `--online-embedding` | Use the configured embedding provider instead of local hash embeddings. |

Query:

```powershell
python scripts\query.py --query "Azure endpoint configuration" --collection docs --top-k 5 --no-rerank
```

Useful query flags:

| Flag | Meaning |
| --- | --- |
| `--query` | User query text. |
| `--top-k` | Number of final results to return. |
| `--collection` | Optional metadata filter. |
| `--verbose` | Print trace and intermediate details. |
| `--no-rerank` | Return fused retrieval results directly. |
| `--online-embedding` | Use configured online embeddings for query vectors. |

Evaluate:

```powershell
python scripts\evaluate.py --test-set tests\fixtures\golden_test_set.json --backend custom --top-k 3
```

## MCP Server

The MCP server uses line-delimited JSON-RPC over stdio. Logs are written to stderr so stdout stays clean for protocol messages.

Available tools:

| Tool | Purpose |
| --- | --- |
| `query_knowledge_hub` | Hybrid search over local indexes, returning Markdown content plus structured citations. |
| `list_collections` | List collections and local asset statistics. |
| `get_document_summary` | Fetch document summary metadata by source name, hash, or doc id. |

Manual stdio smoke test:

```powershell
$env:PYTHONPATH = "src"
'{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m mcp_server.server
```

Example GitHub Copilot `mcp.json`:

```json
{
  "servers": {
    "rag-vibecoding": {
      "type": "stdio",
      "command": "D:\\rag-vibecoding\\RAG-Vibicoding\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "D:\\rag-vibecoding\\RAG-Vibicoding",
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

Example Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-vibecoding": {
      "command": "D:\\rag-vibecoding\\RAG-Vibicoding\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "D:\\rag-vibecoding\\RAG-Vibicoding",
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

Replace the paths if your checkout lives elsewhere.

## Dashboard

Install Streamlit, then launch:

```powershell
python -m pip install streamlit
python scripts\start_dashboard.py
```

Pages:

| Page | What it shows |
| --- | --- |
| Overview | Configured components, data asset counts, trace count, and Dashboard settings. |
| Data Browser | Documents, chunks, metadata, and stored image references. |
| Ingestion Manager | PDF upload ingestion, progress callbacks, and document deletion controls. |
| Ingestion Traces | JSONL ingestion history, stage timing, status, and details. |
| Query Traces | Query history, dense/sparse/fusion/rerank details, and timing waterfall. |
| Evaluation | Golden test set summary, evaluator backend selection, metrics, and case details. |

Dashboard reads the same local `data/` and `logs/traces.jsonl` assets produced by the CLI and MCP tools.

## Running Tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q
pytest -q tests\unit
pytest -q tests\integration
pytest -q tests\e2e
```

Useful targeted checks:

```powershell
pytest -q tests\e2e\test_data_ingestion.py
pytest -q tests\e2e\test_query_cli.py
pytest -q tests\e2e\test_mcp_client.py
pytest -q tests\e2e\test_dashboard_smoke.py
pytest -q tests\e2e\test_recall.py
```

The Streamlit AppTest smoke path is optional and is skipped when `streamlit` is not installed.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `PyYAML is required to read config/settings.yaml` | Run `python -m pip install -r requirements.txt`. |
| `未找到相关文档，请先运行 ingest.py 摄取数据。` | Run `python scripts\ingest.py --path <pdf-or-dir> --collection docs` before querying. |
| Dashboard says Streamlit is required | Run `python -m pip install streamlit`. |
| MCP client cannot import `mcp_server` | Set `PYTHONPATH=src` in the MCP client config or install the project with `python -m pip install -e .`. |
| Online provider returns auth or endpoint errors | Check `api_key`, `azure_endpoint`, `base_url`, `deployment`, and `api_version` in `config/settings.yaml`. |
| Query works but reranking fails | Use `--no-rerank` or set `rerank.backend: none` until the selected reranker dependency/provider is configured. |
| No traces appear in Dashboard | Ensure `observability.enabled: true` and check `observability.log_file`, normally `logs/traces.jsonl`. |

## Current Status

- Completed: scaffold, config loading, pluggable Libs providers, ingestion pipeline, query pipeline, MCP tools, observability, Dashboard pages, evaluation runner, and E2E checks for recall, MCP client, and Dashboard smoke.
- Current progress: Stage I is 2/5 complete; total progress is 65/68.
- Source of truth: [DEV_SPEC.md](DEV_SPEC.md).
