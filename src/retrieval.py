from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.data import Document

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize text with a simple lowercase word tokenizer."""
    return TOKEN_PATTERN.findall(text.lower())


def run_bm25(
    queries: dict[str, str],
    corpus: dict[str, Document],
    top_k: int,
) -> dict[str, list[dict[str, float | str]]]:
    """Retrieve top-k candidate documents for each query using BM25."""
    documents = list(corpus.values())
    tokenized_corpus = [tokenize(document.content) for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    results: dict[str, list[dict[str, float | str]]] = {}
    for query_id, query_text in queries.items():
        scores = bm25.get_scores(tokenize(query_text))
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]
        results[query_id] = [
            {
                "doc_id": documents[index].doc_id,
                "score": float(scores[index]),
            }
            for index in ranked_indexes
        ]

    return results


def save_run(path: str | Path, run: dict[str, list[dict[str, float | str]]]) -> None:
    """Save retrieval or reranking results as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(run, file, indent=2)


def load_run(path: str | Path) -> dict[str, list[dict[str, float | str]]]:
    """Load retrieval or reranking results from JSON."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
