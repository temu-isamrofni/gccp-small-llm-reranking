from __future__ import annotations

from src.data import Document
from src.global_context import build_sentence_anchor
from src.pointwise_reranker import compute_llm_pointwise_score
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


def compute_llm_gccp_score(query: str, document: Document, anchor: str, model, tokenizer, device) -> float:
    """Compute the LLM log-likelihood probability that Passage A is better than Passage B."""
    prompt = build_anchor_comparison_prompt(query, document, anchor)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        
    # Ambil token ID untuk representasi jawaban pilihan "A" atau "B"
    token_A_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    token_B_id = tokenizer.encode("B", add_special_tokens=False)[-1]
    
    target_logits = logits[0, [token_A_id, token_B_id]]
    probs = torch.softmax(target_logits, dim=-1)
    
    return probs[0].item() # Return probabilitas untuk pilihan "A" (Dokumen Kandidat)


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
) -> dict[str, list[dict[str, float | str]]]:
    """Rerank candidates with a real global-context contrastive score using LLM."""
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
            
            # 1. Hitung Pointwise Score Riil menggunakan LLM
            pointwise_score = compute_llm_pointwise_score(query_text, document, model, tokenizer, device)
            
            # 2. Hitung Global Contrastive Score Riil menggunakan LLM terhadap Anchor
            global_score = compute_llm_gccp_score(query_text, document, anchor_info.text, model, tokenizer, device)
            
            # Kombinasi Linear Heterogen sesuai Formulasi Framework GCCP
            score = pointwise_score + anchor_weight * global_score
            scored_documents.append({"doc_id": doc_id, "score": score})

        reranked[query_id] = sorted(
            scored_documents,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return reranked