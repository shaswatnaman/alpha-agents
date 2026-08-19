# Evaluation Framework — AlphaAgents

## Philosophy

We do NOT claim improvements without measured data. This document describes:
1. The evaluation methodology
2. The baseline and experiment configurations
3. How to run experiments
4. Where to find results

---

## Retrieval Evaluation

### Dataset

`eval_datasets/retrieval_eval.json` — a fixed dataset of (query, ticker, relevant_chunk_ids) triples.

To populate this dataset:
1. Ingest at least one document per ticker via `POST /api/v1/documents`
2. For each ticker, write 10-20 queries that have clear answers in the documents
3. Manually review retrieved chunks and label the relevant ones by chunk ID
4. Save to the eval dataset JSON

### Metrics

| Metric | Description |
|---|---|
| Precision@k | Fraction of top-k retrieved chunks that are relevant |
| Recall@k | Fraction of all relevant chunks found in top-k |
| NDCG@k | Normalised Discounted Cumulative Gain — rewards relevant chunks ranked higher |

We use k=6 (our `retrieval_final_k`).

### Experiments

| Configuration | Description |
|---|---|
| `dense_only` | Baseline: dense retrieval only |
| `lexical_only` | Lexical full-text search only |
| `hybrid` | Dense + Lexical + RRF (no MMR) |
| `hybrid_mmr` | Dense + Lexical + RRF + MMR (full pipeline) |

### Running Experiments

```bash
# Run individual experiment
python -m app.evaluation.runner --experiment dense_only
python -m app.evaluation.runner --experiment hybrid_mmr

# Compare all
for exp in dense_only lexical_only hybrid hybrid_mmr; do
  python -m app.evaluation.runner --experiment $exp
done
```

---

## Agent Output Evaluation

### Hallucination Rate

`hallucination_rate` = fraction of FACT-type findings with no evidence IDs.

A finding typed as FACT but citing no evidence is a potential hallucination. Lower is better.

### Citation Accuracy

`citation_accuracy` = |cited chunks ∩ relevant chunks| / |cited chunks|

Requires human-labelled relevant chunk sets. Higher is better.

### Structured Output Validity

Fraction of agent runs that produce valid Pydantic-parseable output. Tracked via Prometheus counter:

```
alphaagents_agent_failures_total{agent="...", reason="..."}
```

A failure rate > 5% suggests the agent's prompt or schema needs revision.

### Agent Agreement Rate

Fraction of agent pairs (fundamental, technical, sentiment) whose `sentiment` values agree. Low agreement signals conflicting evidence or broad market uncertainty — which the Synthesis agent should surface.

---

## Baseline Comparison

### Single-Agent Baseline

A single LLM call receives all context (financial metrics + retrieved evidence + news summary) and produces a free-form research summary. Compared against the multi-agent pipeline on:

- Evidence coverage (how many of the top-20 relevant chunks are cited)
- Citation accuracy (where human labels exist)
- Hallucination rate (FACT findings without evidence)
- Structured output validity (does the output parse cleanly?)
- Latency (seconds wall-clock)
- Estimated LLM cost (USD)

The goal of this comparison is to answer: **"Why does the multi-agent architecture exist?"** — not just to claim it is better, but to quantify the tradeoffs (it is more expensive and slower; it should provide better evidence coverage and citation accuracy in return).

---

## Running the Full Evaluation Suite

```bash
# 1. Ensure database is populated with test documents
# 2. Run retrieval evaluation
python -m app.evaluation.runner --experiment hybrid_mmr

# 3. Inspect Prometheus metrics for agent failure rates
curl http://localhost:8000/metrics | grep alphaagents_agent_failures

# 4. Check research report quality manually for a sample ticker
curl -H "X-API-Key: dev-key-1" http://localhost:8000/api/v1/research/{id}/evidence
```

Results should be logged with experiment_name and stored in `evaluation_results` PostgreSQL table.
