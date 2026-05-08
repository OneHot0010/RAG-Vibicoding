"""Run golden-set evaluation for the local RAG indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.settings import load_settings
from libs.evaluator import EvaluatorFactory
from observability.evaluation import EvalRunner

from query import build_query_components


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        test_set_path = args.test_set or settings.evaluation.golden_test_set
        components = build_query_components(settings, args.data_dir, offline_embedding=not args.online_embedding)
        evaluator = EvaluatorFactory.create(settings, backend=args.backend)
        report = EvalRunner(
            settings,
            hybrid_search=components.hybrid_search,
            evaluator=evaluator,
            top_k=args.top_k,
        ).run(test_set_path)
    except Exception as exc:
        print(f"evaluate failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against a golden test set.")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings YAML path.")
    parser.add_argument("--data-dir", default="data", help="Root directory containing local indexes.")
    parser.add_argument("--test-set", help="Golden test set JSON path. Defaults to settings.evaluation.golden_test_set.")
    parser.add_argument("--backend", help="Optional evaluator backend override.")
    parser.add_argument("--top-k", type=int, help="Override retrieval top_k for every case.")
    parser.add_argument(
        "--online-embedding",
        action="store_true",
        help="Use the embedding provider from settings instead of the offline deterministic backend.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
