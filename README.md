# Evaluating Global-Context Pointwise LLM Reranking under Small Language Models

This repository contains the initial implementation for an Information Retrieval research project based on the SIGIR 2025 paper:

**GCCP: Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information**

The project investigates whether global-context pointwise reranking remains useful when the reranker uses a smaller instruction-tuned language model under limited computational resources.

## Code Provenance

The current code in this repository is an original starter implementation created from scratch for our course progress report. It is not copied from the official GCCP repository.

At this stage, the code provides a lightweight experimental scaffold:

- BM25 retrieval
- placeholder pointwise reranking
- placeholder GCCP-style global-context reranking
- nDCG@k and MRR@k evaluation
- a tiny sample dataset for smoke testing

The official GCCP implementation will be used as a methodological reference for the next development stage.

## Research Question

Can GCCP/PAGC improve zero-shot pointwise reranking when using small 7B/8B language models under limited compute settings?

## Background

Standard pointwise LLM reranking scores each query-document pair independently. This makes the method efficient, but it ignores useful global information from the candidate document set. GCCP addresses this limitation by constructing global context from candidate documents and applying post-aggregated contrastive scoring.

However, it remains unclear whether this strategy is still effective when the reranker uses smaller language models in constrained environments such as free Kaggle notebooks or limited local GPUs.

## Planned Methods

We compare the following methods:

1. BM25 baseline
2. Standard pointwise LLM reranking
3. GCCP/PAGC reranking with global context

## Initial Dataset Plan

The first experiment will use a small BEIR dataset such as **SciFact** because it is suitable for limited compute settings. The current repository also includes a tiny toy dataset under `data/sample/` so the pipeline can be tested without downloading external datasets.

## Evaluation Metrics

We plan to evaluate both effectiveness and efficiency:

- nDCG@10
- MRR@10
- Runtime
- Number of LLM calls
- Approximate token usage

## Repository Structure

```text
.
├── configs/
│   ├── sample.yaml
│   └── scifact.yaml
├── data/
│   └── sample/
├── paper/
│   └── main.tex
├── results/
│   └── README.md
├── scripts/
│   ├── evaluate.py
│   ├── run_bm25.py
│   ├── run_gccp.py
│   └── run_pointwise.py
└── src/
    ├── config.py
    ├── data.py
    ├── evaluation.py
    ├── gccp_reranker.py
    ├── pointwise_reranker.py
    └── retrieval.py
```

## Current Progress

- Selected the main research paper and project direction.
- Defined the research question and experimental scope.
- Prepared the initial repository structure.
- Implemented a minimal BM25 retrieval pipeline.
- Added skeleton pointwise and GCCP reranking modules.
- Added evaluation utilities for nDCG@k and MRR@k.
- Prepared an initial ACL-style paper draft.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

Run BM25 on the sample dataset:

```bash
python scripts/run_bm25.py --config configs/sample.yaml
```

Run the placeholder pointwise reranker:

```bash
python scripts/run_pointwise.py --config configs/sample.yaml
```

Run the placeholder GCCP reranker:

```bash
python scripts/run_gccp.py --config configs/sample.yaml
```

Evaluate a run file:

```bash
python scripts/evaluate.py --config configs/sample.yaml --run results/runs/sample_bm25.json
```

## Notes

The current reranking modules use deterministic lexical scoring as placeholders. They are designed to be replaced with real LLM scoring functions after the baseline pipeline is stable.

## References

- Paper: [Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information](https://arxiv.org/abs/2506.10859)
- Official GitHub repository: [ChainsawM/GCCP](https://github.com/ChainsawM/GCCP)
- Venue: SIGIR 2025
- Authors: Kehan Long, Shasha Li, Chen Xu, Jintao Tang, and Ting Wang

## Next Steps

- Integrate BEIR SciFact loading.
- Replace placeholder scoring with small LLM pointwise scoring.
- Implement GCCP/PAGC anchor construction and contrastive scoring.
- Run experiments with BM25, pointwise reranking, and GCCP/PAGC reranking.
- Analyze effectiveness and efficiency trade-offs.
