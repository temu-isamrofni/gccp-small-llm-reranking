from __future__ import annotations

from src.data import Document
from src.retrieval import tokenize
import torch

YES_NO_PROMPT_TEMPLATE = (
    "Passage: {document}\n"
    "Query: {query}\n"
    "Is the passage relevant to the query? Answer 'yes' or 'no'."
)


def build_pointwise_prompt(query: str, document: Document) -> str:
    """Build the pointwise yes/no relevance prompt used by the planned LLM scorer."""
    return YES_NO_PROMPT_TEMPLATE.format(query=query, document=document.content)


def compute_llm_pointwise_score(query: str, document: Document, model, tokenizer, device) -> float:
    """Compute the model's log-likelihood probability for the 'yes' token."""
    prompt = build_pointwise_prompt(query, document)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :] # Ambil logit dari posisi token terakhir
        
    # Cari id token untuk "yes" dan "no" (pastikan case-insensitive / sesuai perilaku tokenizer)
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[-1]
    
    # Hitung probabilitas ternormalisasi menggunakan Softmax (Metode Peak Relevance)
    target_logits = logits[0, [yes_token_id, no_token_id]]
    probs = torch.softmax(target_logits, dim=-1)
    
    return probs[0].item() # Return probabilitas untuk token "yes"


def rerank_pointwise(
    queries: dict[str, str],
    corpus: dict[str, Document],
    candidates: dict[str, list[dict[str, float | str]]],
    top_k: int,
    model,
    tokenizer,
    device,
) -> dict[str, list[dict[str, float | str]]]:
    """Rerank candidate documents using a real LLM pointwise scorer."""
    reranked: dict[str, list[dict[str, float | str]]] = {}

    for query_id, query_text in queries.items():
        scored_documents: list[dict[str, float | str]] = []
        for candidate in candidates.get(query_id, []):
            doc_id = str(candidate["doc_id"])
            
            # MENGGUNAKAN SKOR LLM RIIL
            score = compute_llm_pointwise_score(query_text, corpus[doc_id], model, tokenizer, device)
            scored_documents.append({"doc_id": doc_id, "score": score})

        reranked[query_id] = sorted(
            scored_documents,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return reranked