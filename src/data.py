from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        """Return the text used by retrieval and reranking modules."""
        return f"{self.title} {self.text}".strip()


def load_queries(path: str | Path) -> dict[str, str]:
    """Load queries from a TSV file with query_id and text columns."""
    queries: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            query_id, query_text = line.split("\t", maxsplit=1)
            queries[query_id] = query_text
    return queries


def load_corpus(path: str | Path) -> dict[str, Document]:
    """Load a JSONL corpus with doc_id, title, and text fields."""
    corpus: dict[str, Document] = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            document = Document(
                doc_id=record["doc_id"],
                title=record.get("title", ""),
                text=record.get("text", ""),
            )
            corpus[document.doc_id] = document
    return corpus


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load relevance judgments from a TSV file."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8") as file:
        header = file.readline().strip().split("\t")
        expected_header = ["query_id", "doc_id", "relevance"]
        if header != expected_header:
            raise ValueError(f"Expected qrels header {expected_header}, got {header}")

        for line in file:
            line = line.strip()
            if not line:
                continue
            query_id, doc_id, relevance = line.split("\t")
            qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    return qrels
