from __future__ import annotations

import math


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Compute discounted cumulative gain at rank k."""
    return sum(
        relevance / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances[:k])
    )


def ndcg_at_k(
    ranked_doc_ids: list[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    """Compute normalized discounted cumulative gain at rank k."""
    gains = [qrels.get(doc_id, 0) for doc_id in ranked_doc_ids]
    ideal_gains = sorted(qrels.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_gains, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def mrr_at_k(
    ranked_doc_ids: list[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    """Compute reciprocal rank for the first relevant document up to rank k."""
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def evaluate_run(
    run: dict[str, list[dict[str, float | str]]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a run with mean nDCG@k and MRR@k."""
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []

    for query_id, query_qrels in qrels.items():
        ranked_doc_ids = [
            str(item["doc_id"])
            for item in run.get(query_id, [])
        ]
        ndcg_scores.append(ndcg_at_k(ranked_doc_ids, query_qrels, k))
        mrr_scores.append(mrr_at_k(ranked_doc_ids, query_qrels, k))

    if not qrels:
        return {f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0}

    return {
        f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
        f"mrr@{k}": sum(mrr_scores) / len(mrr_scores),
    }
