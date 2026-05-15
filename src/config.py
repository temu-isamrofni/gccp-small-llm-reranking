from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    queries_path: Path
    corpus_path: Path
    qrels_path: Path


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int


@dataclass(frozen=True)
class RerankingConfig:
    top_k: int
    anchor_top_k: int


@dataclass(frozen=True)
class OutputConfig:
    run_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    outputs: OutputConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    dataset = raw["dataset"]
    retrieval = raw["retrieval"]
    reranking = raw["reranking"]
    outputs = raw["outputs"]

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset=DatasetConfig(
            name=dataset["name"],
            queries_path=Path(dataset["queries_path"]),
            corpus_path=Path(dataset["corpus_path"]),
            qrels_path=Path(dataset["qrels_path"]),
        ),
        retrieval=RetrievalConfig(top_k=int(retrieval["top_k"])),
        reranking=RerankingConfig(
            top_k=int(reranking["top_k"]),
            anchor_top_k=int(reranking["anchor_top_k"]),
        ),
        outputs=OutputConfig(run_dir=Path(outputs["run_dir"])),
    )
