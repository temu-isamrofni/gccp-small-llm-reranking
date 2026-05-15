from __future__ import annotations

import json
from pathlib import Path

Run = dict[str, list[dict[str, float | str]]]


def save_json_run(path: str | Path, run: Run) -> None:
    """Save retrieval or reranking results as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(run, file, indent=2)


def load_json_run(path: str | Path) -> Run:
    """Load retrieval or reranking results from JSON."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_trec_run(path: str | Path, run: Run, tag: str) -> None:
    """Save a run in standard TREC format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for query_id, ranked_docs in run.items():
            for rank, item in enumerate(ranked_docs, start=1):
                doc_id = str(item["doc_id"])
                score = float(item["score"])
                file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def load_trec_run(path: str | Path) -> Run:
    """Load a standard TREC run file."""
    run: Run = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            query_id, _, doc_id, _, score, *_ = line.split()
            run.setdefault(query_id, []).append(
                {"doc_id": doc_id, "score": float(score)}
            )
    return run


def save_run(path: str | Path, run: Run, tag: str = "RUN") -> None:
    """Save a run as JSON or TREC based on the file extension."""
    output_path = Path(path)
    if output_path.suffix == ".json":
        save_json_run(output_path, run)
    else:
        save_trec_run(output_path, run, tag)


def load_run(path: str | Path) -> Run:
    """Load a run from JSON or TREC format based on the file extension."""
    input_path = Path(path)
    if input_path.suffix == ".json":
        return load_json_run(input_path)
    return load_trec_run(input_path)
