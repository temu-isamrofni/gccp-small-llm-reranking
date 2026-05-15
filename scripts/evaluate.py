from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_qrels
from src.evaluation import evaluate_run
from src.retrieval import load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval or reranking run.")
    parser.add_argument("--config", default="configs/sample.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run = load_run(args.run)
    qrels = load_qrels(config.dataset.qrels_path)
    metrics = evaluate_run(run, qrels, k=args.k)

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
