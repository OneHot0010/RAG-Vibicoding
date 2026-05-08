"""E2E tests for the offline ingestion CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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


def run_ingest(
    repo_root: Path,
    pdf_path: Path,
    data_dir: Path,
    force: bool = False,
    trace_log_file: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/ingest.py",
        "--path",
        str(pdf_path),
        "--collection",
        "e2e",
        "--data-dir",
        str(data_dir),
    ]
    if force:
        command.append("--force")
    if trace_log_file is not None:
        command.extend(["--trace-log-file", str(trace_log_file)])
    return subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)


def test_ingest_cli_builds_local_outputs_and_skips_unchanged_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "sample.pdf"
    data_dir = tmp_path / "data"
    write_minimal_pdf(pdf_path, "Alpha beta retrieval document")

    first = run_ingest(repo_root, pdf_path, data_dir)
    second = run_ingest(repo_root, pdf_path, data_dir)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    assert first_payload["ingested"] == 1
    assert first_payload["skipped"] == 0
    assert second_payload["ingested"] == 0
    assert second_payload["skipped"] == 1
    assert (data_dir / "db" / "ingestion_history.db").is_file()
    assert (data_dir / "db" / "bm25" / "index.json").is_file()
    assert (data_dir / "db" / "chroma" / "records.json").is_file()
    assert (data_dir / "db" / "image_index.db").is_file()


def test_ingest_cli_force_reruns_processed_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "sample.pdf"
    data_dir = tmp_path / "data"
    write_minimal_pdf(pdf_path, "Force rerun document")

    assert run_ingest(repo_root, pdf_path, data_dir).returncode == 0
    forced = run_ingest(repo_root, pdf_path, data_dir, force=True)

    assert forced.returncode == 0, forced.stderr
    payload = json.loads(forced.stdout)
    assert payload["ingested"] == 1
    assert payload["skipped"] == 0


def test_ingest_cli_writes_ingestion_trace_jsonl(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = tmp_path / "trace.pdf"
    data_dir = tmp_path / "data"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    write_minimal_pdf(pdf_path, "Trace ingestion records pipeline stages")

    completed = run_ingest(repo_root, pdf_path, data_dir, trace_log_file=trace_log)

    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in trace_log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trace_type"] == "ingestion"
    assert rows[0]["finished_at"] is not None
    stage_names = [stage["name"] for stage in rows[0]["stages"]]
    assert "ingestion.start" in stage_names
    assert "pipeline.start" in stage_names
    assert "pipeline.completed" in stage_names
    assert "ingestion.completed" in stage_names


def test_ingest_cli_rejects_invalid_input_before_trace_creation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bad_path = tmp_path / "not_pdf.txt"
    data_dir = tmp_path / "data"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    bad_path.write_text("not a pdf", encoding="utf-8")

    completed = run_ingest(repo_root, bad_path, data_dir, trace_log_file=trace_log)

    assert completed.returncode == 1
    assert not trace_log.exists()
