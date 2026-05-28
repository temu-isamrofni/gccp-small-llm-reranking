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


def compute_llm_pointwise_scores_batch(query: str, documents: list[Document], model, tokenizer, device) -> list[float]:
    """Compute the model's log-likelihood probabilities for the 'yes' token across a batch."""
    if not documents:
        return []

    # Konfigurasi left-padding wajib untuk batching decoder-only LLM
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [build_pointwise_prompt(query, doc) for doc in documents]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Berkat left-padding, token jawaban target berada tepat di indeks akhir -1 untuk seluruh baris
        logits = outputs.logits[:, -1, :] 
        
    # Cari id token untuk "yes" dan "no"
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[-1]
    
    # Ambil logit target dan hitung Softmax secara paralel sepanjang dimensi kolom kriteria (dim=-1)
    target_logits = logits[:, [yes_token_id, no_token_id]]
    probs = torch.softmax(target_logits, dim=-1)
    
    return probs[:, 0].tolist() # Mengembalikan list probabilitas token "yes"


def rerank_pointwise(
    queries: dict[str, str],
    corpus: dict[str, Document],
    candidates: dict[str, list[dict[str, float | str]]],
    top_k: int,
    model,
    tokenizer,
    device,
    batch_size: int = 4, # Memproses langsung 20 kandidat sekaligus per kueri
) -> dict[str, list[dict[str, float | str]]]:
    """Rerank candidate documents using a batched LLM pointwise scorer."""
    reranked: dict[str, list[dict[str, float | str]]] = {}

    for query_id, query_text in queries.items():
        query_candidates = candidates.get(query_id, [])
        if not query_candidates:
            continue

        doc_ids = [str(cand["doc_id"]) for cand in query_candidates]
        documents = [corpus[doc_id] for doc_id in doc_ids]

        all_scores: list[float] = []
        # Pembagian chunk dokumen berdasarkan ukuran batch_size
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_scores = compute_llm_pointwise_scores_batch(
                query_text, batch_docs, model, tokenizer, device
            )
            all_scores.extend(batch_scores)

        scored_documents = [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in zip(doc_ids, all_scores)
        ]

        reranked[query_id] = sorted(
            scored_documents,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    return reranked