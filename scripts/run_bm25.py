from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_corpus, load_queries
from src.retrieval import run_bm25, save_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval.")
    parser.add_argument("--config", default="configs/sample.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    queries = load_queries(config.dataset.queries_path)
    corpus = load_corpus(config.dataset.corpus_path)
    run = run_bm25(queries, corpus, config.retrieval.top_k)

    output_path = config.outputs.run_dir / f"{config.experiment_name}_bm25.json"
    save_run(output_path, run)
    print(f"Saved BM25 run to {output_path}")


if __name__ == "__main__":
    main()
