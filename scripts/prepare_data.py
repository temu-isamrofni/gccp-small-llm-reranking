import json
from datasets import load_dataset
from pathlib import Path

def prepare_scifact():
    out_dir = Path("data/beir/scifact")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Memuat dataset dari mteb yang sudah terbukti aman dari error loading script
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    default_ds = load_dataset("mteb/scifact", "default", split="test") # qrels

    # 1. Simpan Corpus ke jsonl
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for item in corpus_ds:
            f.write(json.dumps({"doc_id": item["_id"], "title": item["title"], "text": item["text"]}) + "\n")
            
    # 2. Simpan Queries ke tsv
    with open(out_dir / "queries.tsv", "w", encoding="utf-8") as f:
        for item in queries_ds:
            f.write(f"{item['_id']}\t{item['text']}\n")
            
    # 3. Simpan Qrels ke tsv (PERBAIKAN: Menggunakan tanda hubung '-' sesuai skema MTEB)
    with open(out_dir / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\trelevance\n")
        for item in default_ds:
            f.write(f"{item['query-id']}\t{item['corpus-id']}\t{item['score']}\n")

if __name__ == "__main__":
    prepare_scifact()