"""E2E recall regression test using the golden evaluation runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


MIN_HIT_RATE = 1.0


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


def run_command(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def write_golden(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "azure endpoint deployment api key",
                        "expected_sources": ["azure_recall.pdf"],
                        "filters": {"collection": "recall"},
                        "top_k": 3,
                    },
                    {
                        "query": "hybrid dense sparse fusion retrieval",
                        "expected_sources": ["hybrid_recall.pdf"],
                        "filters": {"collection": "recall"},
                        "top_k": 3,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_recall_regression_meets_hit_rate_threshold(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = tmp_path / "docs"
    data_dir = tmp_path / "data"
    golden = tmp_path / "golden_recall.json"
    docs_dir.mkdir()
    write_minimal_pdf(
        docs_dir / "azure_recall.pdf",
        "Azure endpoint deployment api key configuration guide for recall regression",
    )
    write_minimal_pdf(
        docs_dir / "hybrid_recall.pdf",
        "Hybrid retrieval combines dense sparse fusion ranking for recall regression",
    )
    write_golden(golden)

    ingest = run_command(
        repo_root,
        [
            "scripts/ingest.py",
            "--path",
            str(docs_dir),
            "--collection",
            "recall",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert ingest.returncode == 0, ingest.stderr

    evaluation = run_command(
        repo_root,
        [
            "scripts/evaluate.py",
            "--data-dir",
            str(data_dir),
            "--test-set",
            str(golden),
            "--backend",
            "custom",
            "--top-k",
            "3",
        ],
    )

    assert evaluation.returncode == 0, evaluation.stderr
    payload = json.loads(evaluation.stdout)
    assert payload["case_count"] == 2
    assert payload["metrics"]["hit_rate"] >= MIN_HIT_RATE
    assert all(case["metrics"]["hit_rate"] == 1.0 for case in payload["cases"])
