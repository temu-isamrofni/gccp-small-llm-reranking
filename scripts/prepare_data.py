import json
from datasets import load_dataset
from pathlib import Path

def prepare_scifact():
    out_dir = Path("data/beir/scifact")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Memuat dataset berformat Parquet mteb
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    default_ds = load_dataset("mteb/scifact", "default", split="test") 

    # Simpan Corpus ke jsonl
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for item in corpus_ds:
            f.write(json.dumps({"doc_id": item["_id"], "title": item["title"], "text": item["text"]}) + "\n")
            
    # Simpan Queries ke tsv
    with open(out_dir / "queries.tsv", "w", encoding="utf-8") as f:
        for item in queries_ds:
            f.write(f"{item['_id']}\t{item['text']}\n")
            
    # Simpan Qrels ke tsv
    with open(out_dir / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\trelevance\n")
        for item in default_ds:
            f.write(f"{item['query_id']}\t{item['corpus_id']}\t{item['score']}\n")

if __name__ == "__main__":
    prepare_scifact()