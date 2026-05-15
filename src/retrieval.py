from __future__ import annotations

import re

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
