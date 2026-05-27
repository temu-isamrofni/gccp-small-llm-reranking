from __future__ import annotations

import re
from dataclasses import dataclass

from src.data import Document
from src.retrieval import tokenize

import numpy as np
from scipy.linalg import eigh
from sklearn.feature_extraction.text import TfidfVectorizer

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

def build_sentence_anchor_spectral(query, corpus, candidates, anchor_top_k, z=10, theta=0.1):
    # 1. Kumpulkan semua kalimat dari top-m candidates
    selected_docs = candidates[:anchor_top_k]
    sentences = []
    sentence_metadata = [] # untuk melacak doc_id asli dan urutan
    
    for rank, cand in enumerate(selected_docs):
        doc = corpus[str(cand["doc_id"])]
        cand_sentences = split_sentences(doc.content)
        for idx, sent in enumerate(cand_sentences):
            sentences.append(sent)
            sentence_metadata.append({"doc_id": cand["doc_id"], "orig_idx": idx, "text": sent})
            
    if not sentences:
        return AnchorInfo("", [], [])

    # 2. Konstruksi Afinitas Matrix (A) menggunakan TF-IDF Cosine Similarity
    vec = TfidfVectorizer()
    tfidf_matrix = vec.fit_transform(sentences).toarray()
    A = np.dot(tfidf_matrix, tfidf_matrix.T)
    A[A < theta] = 0.0  # Thresholding filter noise
    np.fill_diagonal(A, 0)
    
    # 3. Hitung Matriks Derajat (D) dan Laplacian Terormalisasi (L)
    d = np.sum(A, axis=1)
    d[d == 0] = 1e-6 # proteksi pembagian nol
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    I = np.eye(len(sentences))
    L = I - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
    
    # 4. Selesaikan Persamaan Nilai Eigen untuk mendapatkan Vektor Fiedler (v_2)
    eigenvalues, eigenvectors = eigh(L)
    # Ambil indeks terurut, nilai eigen terkecil kedua biasanya berada pada indeks 1
    fiedler_vector = eigenvectors[:, 1]
    
    # 5. Klasterisasi berdasarkan tanda komponen Fiedler Vector
    cluster_pos = [sentence_metadata[i] for i in range(len(sentences)) if fiedler_vector[i] >= 0]
    cluster_neg = [sentence_metadata[i] for i in range(len(sentences)) if fiedler_vector[i] < 0]
    chosen_cluster = cluster_pos if len(cluster_pos) >= len(cluster_neg) else cluster_neg
    
    # 6. Urutkan kembali demi koherensi diskursus teks
    chosen_cluster.sort(key=lambda x: (x["doc_id"], x["orig_idx"]))
    selected_sentences = [item["text"] for item in chosen_cluster[:z]]
    
    return AnchorInfo(
        text=" ".join(selected_sentences),
        source_doc_ids=list(set([item["doc_id"] for item in chosen_cluster])),
        selected_sentences=selected_sentences
    )