from __future__ import annotations

from src.run_io import Run


def normalize_query_scores(items: list[dict[str, float | str]]) -> dict[str, float]:
    """Normalize scores for one query with min-max normalization."""
    if not items:
        return {}

    scores = [float(item["score"]) for item in items]
    min_score = min(scores)
    max_score = max(scores)
    denominator = max_score - min_score

    normalized: dict[str, float] = {}
    for item in items:
        doc_id = str(item["doc_id"])
        score = float(item["score"])
        if denominator == 0.0:
            normalized[doc_id] = 0.0
        else:
            normalized[doc_id] = (score - min_score) / denominator
    return normalized


def aggregate_pagc(
    pointwise_run: Run,
    gccp_run: Run,
    top_k: int,
    pointwise_weight: float = 0.5,
    gccp_weight: float = 0.5,
) -> Run:
    """
    Combine pointwise and GCCP scores with normalized linear aggregation.

    This mirrors the PAGC idea at a starter-project level: pointwise relevance
    scores are post-aggregated with global-context contrastive scores.
    """
    aggregated: Run = {}
    query_ids = sorted(set(pointwise_run) | set(gccp_run))

    for query_id in query_ids:
        pointwise_scores = normalize_query_scores(pointwise_run.get(query_id, []))
        gccp_scores = normalize_query_scores(gccp_run.get(query_id, []))
        doc_ids = set(pointwise_scores) | set(gccp_scores)

        ranked_docs = [
            {
                "doc_id": doc_id,
                "score": (
                    pointwise_weight * pointwise_scores.get(doc_id, 0.0)
                    + gccp_weight * gccp_scores.get(doc_id, 0.0)
                ),
            }
            for doc_id in doc_ids
        ]
        aggregated[query_id] = sorted(
            ranked_docs,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return aggregated
