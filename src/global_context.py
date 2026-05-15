from __future__ import annotations

import re
from dataclasses import dataclass

from src.data import Document
from src.retrieval import tokenize

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class AnchorInfo:
    """Container for a generated global anchor and basic diagnostics."""

    text: str
    source_doc_ids: list[str]
    selected_sentences: list[str]


def split_sentences(text: str) -> list[str]:
    """Split text into lightweight sentence candidates."""
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(text)
        if sentence.strip()
    ]
    return sentences if sentences else [text.strip()]


def sentence_score(sentence: str, candidate_terms: set[str], query_terms: set[str]) -> float:
    """Score a sentence by candidate-set coverage and query overlap."""
    terms = set(tokenize(sentence))
    if not terms:
        return 0.0
    coverage = len(terms & candidate_terms) / max(len(terms), 1)
    query_overlap = len(terms & query_terms) / max(len(query_terms), 1)
    return coverage + query_overlap


def build_sentence_anchor(
    query: str,
    corpus: dict[str, Document],
    candidates: list[dict[str, float | str]],
    anchor_top_k: int,
    max_sentences: int = 5,
) -> AnchorInfo:
    """
    Build a global anchor from representative sentences in top candidate documents.

    This is a deterministic approximation of the GCCP global context construction
    step. The official method uses multi-document summarization; this starter
    implementation keeps the same interface without requiring heavy dependencies.
    """
    selected_candidates = candidates[:anchor_top_k]
    source_doc_ids = [str(candidate["doc_id"]) for candidate in selected_candidates]
    query_terms = set(tokenize(query))

    candidate_terms: set[str] = set()
    sentence_pool: list[tuple[str, str]] = []
    for doc_id in source_doc_ids:
        document = corpus[doc_id]
        candidate_terms.update(tokenize(document.content))
        for sentence in split_sentences(document.content):
            sentence_pool.append((doc_id, sentence))

    ranked_sentences = sorted(
        sentence_pool,
        key=lambda item: sentence_score(item[1], candidate_terms, query_terms),
        reverse=True,
    )

    seen: set[str] = set()
    selected_sentences: list[str] = []
    for _, sentence in ranked_sentences:
        normalized = " ".join(sentence.lower().split())
        if normalized in seen:
            continue
        selected_sentences.append(sentence)
        seen.add(normalized)
        if len(selected_sentences) >= max_sentences:
            break

    anchor_text = " ".join(selected_sentences)
    return AnchorInfo(
        text=anchor_text,
        source_doc_ids=source_doc_ids,
        selected_sentences=selected_sentences,
    )
