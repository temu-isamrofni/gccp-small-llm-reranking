from __future__ import annotations

from src.data import Document
from src.global_context import build_sentence_anchor
from src.pointwise_reranker import lexical_pointwise_score
from src.retrieval import tokenize

COMPARE_WITH_ANCHOR_PROMPT_TEMPLATE = (
    'Given a query "{query}", which of the following two passages is more relevant '
    'to the query?\n\n'
    'Passage A: "{document}"\n\n'
    'Passage B: "{anchor}"\n\n'
    "Output Passage A or Passage B."
)


def build_anchor_comparison_prompt(query: str, document: Document, anchor: str) -> str:
    """Build the planned document-vs-anchor comparison prompt for GCCP scoring."""
    return COMPARE_WITH_ANCHOR_PROMPT_TEMPLATE.format(
        query=query,
        document=document.content,
        anchor=anchor,
    )


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
        anchor_info = build_sentence_anchor(
            query=query_text,
            corpus=corpus,
            candidates=query_candidates,
            anchor_top_k=anchor_top_k,
        )
        scored_documents: list[dict[str, float | str]] = []

        for candidate in query_candidates:
            doc_id = str(candidate["doc_id"])
            document = corpus[doc_id]
            pointwise_score = lexical_pointwise_score(query_text, document)
            global_score = anchor_similarity(anchor_info.text, document)
            score = pointwise_score + anchor_weight * global_score
            scored_documents.append({"doc_id": doc_id, "score": score})

        reranked[query_id] = sorted(
            scored_documents,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return reranked
