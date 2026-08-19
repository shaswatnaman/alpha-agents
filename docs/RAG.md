# RAG Pipeline — AlphaAgents

## Why RAG?

Financial research requires grounding in source documents — annual reports, SEC filings, earnings transcripts. Without retrieval, an LLM makes claims based on training data that may be stale, incomplete, or wrong for the specific company being researched.

RAG lets the LLM answer questions about a specific company's current state using documents you control.

---

## Embeddings

An embedding is a fixed-size vector of floats that represents the semantic meaning of a piece of text. Texts with similar meaning have vectors that are close together in embedding space (measured by cosine similarity).

We use OpenAI `text-embedding-3-small` (1536 dimensions) for two reasons:
1. It is cost-effective for the document volumes we handle
2. It supports batch embedding (up to 100 texts per API call)

---

## Chunking Strategy

Documents are split into overlapping fixed-size chunks (512 chars, 64-char overlap) with sentence-boundary awareness:

```
document text
    → split at ~512 chars
    → extend to nearest sentence boundary (. ! ?)
    → add 64-char overlap with next chunk
```

**Why overlap?** Facts often span sentence boundaries. If "Revenue increased 20%" ends a sentence at position 510 and the next chunk starts at 512, the claim is split across chunks. Overlap ensures that boundary-spanning facts appear fully in at least one chunk.

**Why fixed-size over semantic chunking?** Semantic chunking uses an LLM to decide split points — slower and costlier. Fixed-size with sentence extension achieves 90% of the quality at 10% of the cost.

---

## Dense Retrieval

Query text is embedded, then compared against all stored chunk embeddings using cosine similarity via pgvector:

```sql
ORDER BY embedding <=> query_embedding
```

This captures **semantic** matches — "profit declined" matches "earnings fell" even though they share no words.

**Failure mode:** Dense retrieval misses exact-match keywords. A query for "RSI-14" may not match "14-day relative strength index" because the embedding space compresses them differently.

---

## Lexical Retrieval

PostgreSQL full-text search (`tsvector`/`tsquery`) with `ts_rank_cd` scoring. Captures exact keyword matches that dense retrieval misses.

```sql
WHERE text_tsv @@ websearch_to_tsquery('english', query)
ORDER BY ts_rank_cd(text_tsv, ...) DESC
```

**Failure mode:** Lexical retrieval misses paraphrasing. "Gross margin compressed" does not match a query for "lower profitability."

---

## Hybrid Retrieval via RRF

Reciprocal Rank Fusion (Cormack et al., 2009) combines the two ranked lists without requiring score normalisation:

```
RRF_score(chunk) = Σ 1/(k + rank)  across all ranked lists, k=60
```

A chunk that ranks 3rd in dense and 5th in lexical scores higher than one that ranks 1st only in lexical. This handles the complementary failure modes of each individual retriever.

---

## MMR Reranking

Maximum Marginal Relevance (Carbonell & Goldstein, 1998) selects the final top-k chunks iteratively to maximise both relevance and diversity:

```
score(chunk) = λ · relevance(chunk, query) − (1−λ) · max_similarity(chunk, selected)
```

With λ=0.6, we balance 60% relevance against 40% diversity pressure.

**Why MMR?** Without it, the top-6 chunks are often near-identical paragraphs from the same section of an annual report. The LLM receives redundant context and hallucinates gaps. MMR ensures the 6 selected chunks cover different aspects of the query.

---

## Retrieval Failure Modes and Mitigations

| Failure | Mitigation |
|---|---|
| No relevant documents | Return empty evidence; agent marks claim as UNCERTAINTY |
| Stale documents | Metadata filter by `published_date`; agent flags recency risk |
| Irrelevant chunks retrieved | MMR diversity + evidence quality threshold |
| Hallucinated claims | Evidence-grounded output requirement; Critic agent checks |
| Chunking splits a key sentence | Overlapping chunks ensure each sentence appears whole |

---

## Evaluation

We maintain a fixed retrieval evaluation dataset (`eval_datasets/retrieval_eval.json`) with:
- 20 queries per ticker
- Human-labelled relevant chunk IDs

Metrics: Precision@6, Recall@6, NDCG@6

Experiments:
1. Dense only (baseline)
2. Lexical only
3. Dense + Lexical + RRF (no MMR)
4. Dense + Lexical + RRF + MMR (full pipeline)

See `docs/EVALUATION.md` for results.
