from __future__ import annotations

from collections import Counter

from src.data import Document
from src.pointwise_reranker import lexical_pointwise_score
from src.retrieval import tokenize


def build_global_anchor(
    corpus: dict[str, Document],
    candidates: list[dict[str, float | str]],
    anchor_top_k: int,
) -> str:
    """
    Build a simple lexical anchor from the highest-ranked candidate documents.

    This is a lightweight placeholder for the GCCP global context construction
    step. A future version should replace this with an LLM-generated anchor.
    """
    term_counts: Counter[str] = Counter()
    for candidate in candidates[:anchor_top_k]:
        doc_id = str(candidate["doc_id"])
        term_counts.update(tokenize(corpus[doc_id].content))

    anchor_terms = [term for term, _ in term_counts.most_common(20)]
    return " ".join(anchor_terms)


def anchor_similarity(anchor: str, document: Document) -> float:
    """Compute a simple overlap score between the anchor and a document."""
    anchor_terms = set(tokenize(anchor))
    document_terms = set(tokenize(document.content))
    if not anchor_terms or not document_terms:
        return 0.0
    return len(anchor_terms & document_terms) / len(anchor_terms)


def rerank_gccp(
    queries: dict[str, str],
    corpus: dict[str, Document],
    candidates: dict[str, list[dict[str, float | str]]],
    top_k: int,
    anchor_top_k: int,
    anchor_weight: float = 0.25,
) -> dict[str, list[dict[str, float | str]]]:
    """
    Rerank candidates with a placeholder global-context score.

    The current score combines a query-document pointwise score with a document
    to global-anchor similarity score. This approximates the intended GCCP
    interface while keeping the initial progress implementation lightweight.
    """
    reranked: dict[str, list[dict[str, float | str]]] = {}

    for query_id, query_text in queries.items():
        query_candidates = candidates.get(query_id, [])
        anchor = build_global_anchor(corpus, query_candidates, anchor_top_k)
        scored_documents: list[dict[str, float | str]] = []

        for candidate in query_candidates:
            doc_id = str(candidate["doc_id"])
            document = corpus[doc_id]
            pointwise_score = lexical_pointwise_score(query_text, document)
            global_score = anchor_similarity(anchor, document)
            score = pointwise_score + anchor_weight * global_score
            scored_documents.append({"doc_id": doc_id, "score": score})

        reranked[query_id] = sorted(
            scored_documents,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return reranked
