# Evidence-Based Question Answering — RAG + Small-LM Pipeline

> **ADM & LT 2025/2026 Hackathon** — Team *Los Pollos Hermanos*
> A query-type-aware retrieval-augmented pipeline that answers natural-language
> questions over a per-database corpus of **synthetic** facts.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/runtime-Kaggle%20%7C%20GPU-orange)

---

## Overview

The task is evidence-based QA: for each question we must derive the answer **only**
from the facts of the referenced evidence database (`db_id`). Every entity name has
been replaced with a synthetic token absent from any pre-training corpus, so the
generator cannot answer from parametric memory — the answer must come from
**retrieval + reasoning over the evidence**.

Five query types must be handled, each evaluated with its own metric:

| Query type    | Answer format                       | Metric        |
|---------------|-------------------------------------|---------------|
| `Lookup`      | single string                       | exact match   |
| `Aggregation` | single string                       | exact match   |
| `Count`       | integer as string (e.g. `3`)        | accuracy      |
| `Boolean`     | `yes` / `no`                        | accuracy      |
| `Set ops`     | JSON list of strings (`["a","b"]`)  | set-based F1  |

The final score is the **unweighted macro-average** across the five types.

## Design principles

1. **No single generic answerer.** Each query type has a different structure, so the
   pipeline routes every question to specialised logic.
2. **Retrieval quality first.** The evidence space is reduced through lexical search,
   dense retrieval, rank fusion and reranking before any generation happens.
3. **Models *and* deterministic rules.** Some queries need flexible text understanding;
   others need strict operations (counting, numeric comparison, controlled-format sets).
   Heuristics run first and a language model is the fallback.

## Architecture

```mermaid
flowchart TD
    Q[Query] --> R{Query-type router<br/>TF-IDF + LogReg}
    Q --> RET

    subgraph RET [Hybrid retrieval]
        BM25[BM25 lexical] --> RRF[Reciprocal Rank Fusion]
        DENSE[BGE-large dense] --> RRF
        RRF --> RERANK[MiniLM cross-encoder rerank]
    end

    R -->|Lookup| L[Qwen2.5-0.5B + LoRA]
    R -->|Count| C[Population / gender heuristics<br/>→ Qwen fallback]
    R -->|Aggregation| A[Numeric extremum + frequency count<br/>→ Qwen → FLAN-T5 fallback]
    R -->|Boolean| B[Gender / membership checks<br/>→ FLAN-T5]
    R -->|Set ops| S[Gender heuristics<br/>→ batched Qwen fallback]

    RERANK --> L & C & A & B & S
    L & C & A & B & S --> OUT[Answer]
```

**Components**

| Stage       | Model / method                              | Size   |
|-------------|---------------------------------------------|--------|
| Router      | TF-IDF (1–2 grams) + Logistic Regression    | —      |
| Dense retr. | `BAAI/bge-large-en-v1.5`                     | 335M   |
| Reranker    | `cross-encoder/ms-marco-MiniLM-L-12-v2`     | 33M    |
| Generator A | `Qwen/Qwen2.5-0.5B-Instruct` + LoRA adapter | 0.5B   |
| Generator B | `google/flan-t5-base`                       | 250M   |

The LoRA adapter is fine-tuned on `Lookup` / `Count` / `Set ops` training examples,
pairing BM25-selected top facts with the gold answer under a query-type-specific
system prompt.

## Repository structure

```
adm-lt-hackathon-qa/
├── README.md
├── LICENSE
├── requirements.txt
├── Notebook_NLP_LOSPOLLOSHERMANOS.ipynb   # end-to-end pipeline
└── report/
    └── LOS_POLLOS_HERMANOS.pdf            # written report
```

The whole system lives in a single, self-contained notebook: data loading,
retrieval, LoRA fine-tuning, dev-set evaluation and submission generation.


## Team

Denise Brancaleone · Matías Cerda · Marco Restuccia · Gianluca Pasquinelli
