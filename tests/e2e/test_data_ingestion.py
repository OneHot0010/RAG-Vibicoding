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


def run_ingest(repo_root: Path, pdf_path: Path, data_dir: Path, force: bool = False) -> subprocess.CompletedProcess[str]:
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
