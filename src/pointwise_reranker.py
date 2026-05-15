from __future__ import annotations

from src.data import Document
from src.retrieval import tokenize

YES_NO_PROMPT_TEMPLATE = (
    "Passage: {document}\n"
    "Query: {query}\n"
    "Is the passage relevant to the query? Answer 'yes' or 'no'."
)


def build_pointwise_prompt(query: str, document: Document) -> str:
    """Build the pointwise yes/no relevance prompt used by the planned LLM scorer."""
    return YES_NO_PROMPT_TEMPLATE.format(query=query, document=document.content)


def lexical_pointwise_score(query: str, document: Document) -> float:
    """
    Compute a deterministic placeholder score for a query-document pair.

    This function will be replaced with an LLM-based pointwise scoring function.
    """
    query_terms = set(tokenize(query))
    document_terms = tokenize(document.content)
    if not query_terms or not document_terms:
        return 0.0

    overlap = sum(1 for term in document_terms if term in query_terms)
    return overlap / len(document_terms)


def rerank_pointwise(
    queries: dict[str, str],
    corpus: dict[str, Document],
    candidates: dict[str, list[dict[str, float | str]]],
    top_k: int,
) -> dict[str, list[dict[str, float | str]]]:
    """Rerank candidate documents using a placeholder pointwise scorer."""
    reranked: dict[str, list[dict[str, float | str]]] = {}

    for query_id, query_text in queries.items():
        scored_documents: list[dict[str, float | str]] = []
        for candidate in candidates.get(query_id, []):
            doc_id = str(candidate["doc_id"])
            score = lexical_pointwise_score(query_text, corpus[doc_id])
            scored_documents.append({"doc_id": doc_id, "score": score})

        reranked[query_id] = sorted(
            scored_documents,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return reranked
