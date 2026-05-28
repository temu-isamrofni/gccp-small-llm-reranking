from __future__ import annotations

from src.data import Document
from src.global_context import build_sentence_anchor
from src.pointwise_reranker import compute_llm_pointwise_scores_batch
import torch

COMPARE_WITH_ANCHOR_PROMPT_TEMPLATE = (
    'Given a query "{query}", which of the following two passages is more relevant '
    'to the query?\n\n'
    'Passage A: "{document}"\n\n'
    'Passage B: "{anchor}"\n\n'
    "Output Passage A or Passage B."
)


def build_anchor_comparison_prompt(query: str, document: Document, anchor: str) -> str:
    """Build the document-vs-anchor comparison prompt for GCCP scoring."""
    return COMPARE_WITH_ANCHOR_PROMPT_TEMPLATE.format(
        query=query,
        document=document.content,
        anchor=anchor,
    )


def compute_llm_gccp_scores_batch(
    query: str, documents: list[Document], anchor: str, model, tokenizer, device
) -> list[float]:
    """Compute the LLM log-likelihood probabilities that Passage A is better than Passage B across a batch."""
    if not documents:
        return []

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [build_anchor_comparison_prompt(query, doc, anchor) for doc in documents]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        
    token_A_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    token_B_id = tokenizer.encode("B", add_special_tokens=False)[-1]
    
    target_logits = logits[:, [token_A_id, token_B_id]]
    probs = torch.softmax(target_logits, dim=-1)
    
    return probs[:, 0].tolist() # Mengembalikan list probabilitas untuk pilihan "A" (Dokumen Kandidat)


def rerank_gccp(
    queries: dict[str, str],
    corpus: dict[str, Document],
    candidates: dict[str, list[dict[str, float | str]]],
    top_k: int,
    anchor_top_k: int,
    model,
    tokenizer,
    device,
    anchor_weight: float = 0.25,
    batch_size: int = 4,
) -> dict[str, list[dict[str, float | str]]]:
    """Rerank candidates with a real global-context contrastive score using batched LLM inference."""
    reranked: dict[str, list[dict[str, float | str]]] = {}

    for query_id, query_text in queries.items():
        query_candidates = candidates.get(query_id, [])
        if not query_candidates:
            continue

        # Ekstraksi dokumen jangkar (Anchor) menggunakan Spectral MDS kustom
        anchor_info = build_sentence_anchor(
            query=query_text,
            corpus=corpus,
            candidates=query_candidates,
            anchor_top_k=anchor_top_k,
        )

        doc_ids = [str(cand["doc_id"]) for cand in query_candidates]
        documents = [corpus[doc_id] for doc_id in doc_ids]

        # 1. Ambil Nilai Pointwise Komprehensif via Batch
        pointwise_scores: list[float] = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            scores = compute_llm_pointwise_scores_batch(
                query_text, batch_docs, model, tokenizer, device
            )
            pointwise_scores.extend(scores)

        # 2. Ambil Nilai Kontrastif Global Context via Batch terhadap Anchor
        global_scores: list[float] = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            scores = compute_llm_gccp_scores_batch(
                query_text, batch_docs, anchor_info.text, model, tokenizer, device
            )
            global_scores.extend(scores)

        # 3. Penggabungan Linear Komponen Skor Sesuai Framework Asli GCCP
        scored_documents: list[dict[str, float | str]] = []
        for doc_id, p_score, g_score in zip(doc_ids, pointwise_scores, global_scores):
            score = p_score + anchor_weight * g_score
            scored_documents.append({"doc_id": doc_id, "score": score})

        reranked[query_id] = sorted(
            scored_documents,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return reranked