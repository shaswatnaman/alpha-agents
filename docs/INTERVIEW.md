# Interview Preparation — AlphaAgents

---

## ARCHITECTURE

**Walk me through the architecture.**

A client sends `POST /api/v1/research` with a ticker. The FastAPI gateway validates the API key, checks rate limits (Redis sliding window), and checks idempotency (Redis + PostgreSQL). If the request is new, a `ResearchRequest` is persisted with status PENDING, and an asyncio background task starts the pipeline. The API responds 202 immediately with a `research_id`.

The pipeline: (1) concurrently fetches technical indicators, financial metrics, and news; (2) runs hybrid RAG retrieval against pgvector; (3) runs Fundamental, Technical, and Sentiment agents concurrently; (4) runs the Critic agent sequentially; (5) runs the Synthesis agent; (6) assembles the final `ResearchReport`, caches it in Redis, and persists it to PostgreSQL.

**Why multi-agent?**

Each domain of financial analysis requires different inputs and reasoning modes. Fundamental analysis needs document-grounded financial metrics. Technical analysis needs deterministic numerical indicators. Sentiment analysis needs recent news. Combining these into one prompt creates a context-pollution problem — the LLM sees irrelevant information for each subtask and its focus degrades.

Separation also enables independent failure handling. If technical data is unavailable, technical analysis fails gracefully while the other two agents proceed. A monolithic agent would fail entirely.

Finally, the Critic agent can only function if it has independent agent outputs to challenge. A single agent cannot critique itself.

**Why not one LLM?**

Passing everything to one prompt creates: (a) a 20,000-token context that the model attends to poorly; (b) no way to attribute which reasoning produced which claim; (c) no independent verification of claims; (d) no partial failure — if one data source fails, the whole prompt breaks. Multi-agent enables isolation, attribution, and redundancy.

**Why FastAPI?**

Async-native, so LLM API calls yield the event loop rather than blocking a thread. Pydantic integration for schema validation at the API boundary. Auto-generated OpenAPI docs. Background task support for non-blocking pipeline execution.

**How does a request flow through the system?**

`POST /api/v1/research` → API key check → rate limit check → idempotency check → create `ResearchRequest` in DB → launch background task → respond 202. Meanwhile: data fetch → RAG retrieval → 3 agents concurrent → critic → synthesis → assemble report → Redis cache → PostgreSQL persist → update status to COMPLETED. Client polls `GET /api/v1/research/{id}` for status, then `GET /api/v1/research/{id}/report` for the result.

---

## RAG

**What is an embedding?**

A fixed-dimension vector of floats that encodes the semantic meaning of a text. Texts about similar topics have vectors that are geometrically close (measured by cosine similarity). We use OpenAI `text-embedding-3-small` which produces 1536-dimensional vectors.

**How does vector search work?**

We store each chunk's embedding in PostgreSQL via pgvector. At query time, we embed the query string and compute cosine similarity against all stored embeddings. pgvector's IVFFlat index clusters the vectors into 100 buckets and searches only the most relevant buckets, making retrieval sub-linear in the number of vectors.

**Why MMR?**

Without MMR, the top-k results are often near-identical paragraphs from the same section of a document. The LLM receives redundant context and may miss important information from other sections. MMR iteratively selects the next chunk that maximises relevance to the query minus similarity to already-selected chunks, producing a diverse, complementary context window.

**Why hybrid retrieval?**

Dense retrieval (embedding similarity) captures semantic matches but misses exact-match keywords. Lexical retrieval (PostgreSQL full-text search) captures exact keywords but misses paraphrasing. Hybrid via RRF gets the benefits of both.

**How do you prevent irrelevant retrieval?**

Metadata filtering (ticker, document_type, date range) narrows the search space. MMR reranking removes redundant results. The Critic agent challenges claims that have no evidence IDs. The confidence score decreases when evidence is sparse.

**How do you evaluate retrieval?**

Fixed evaluation dataset with 20 queries per ticker, human-labelled relevant chunk IDs. Metrics: Precision@6, Recall@6, NDCG@6. We compare four configurations: dense-only (baseline), lexical-only, hybrid-no-MMR, hybrid+MMR.

---

## LLM

**How do structured outputs work?**

OpenAI's `beta.chat.completions.parse` endpoint accepts a Pydantic model as `response_format`. The model is compiled to a JSON Schema and sent to the API. The response is guaranteed to conform to that schema. We then call `Model.model_validate_json(response.content)` to get a typed Python object — no regex parsing.

**How do you handle malformed outputs?**

Pydantic validation raises `ValidationError` if the schema is violated. This is caught in the agent's `try/except`, which returns a `failed=True` AgentReport. The pipeline continues without that agent's findings. The synthesis agent notes the missing input.

**How do you handle hallucinations?**

Three mitigations: (1) Technical indicators are computed in deterministic Python — the LLM never calculates numbers. (2) Agents must reference evidence IDs for FACT claims — claims without evidence are flagged by the Critic. (3) We measure hallucination rate (fraction of FACT findings with no evidence IDs) in our evaluation framework.

**How do you reduce LLM cost?**

Model selection: `gpt-4o-mini` for all five agents at ~10× lower cost than GPT-4. Structured output avoids long follow-up calls for parsing. Token usage is tracked per agent and per run. The research result is cached in Redis for 1 hour — repeat reads of the same report cost zero LLM tokens.

---

## AGENTS

**What makes an agent different from an LLM call?**

An agent has: a defined role and responsibility boundary, a specific prompt encoding that role, structured output requirements, access to domain-appropriate context (not generic context), and independent failure handling. An LLM call is a function call; an agent is a cohesive unit with its own interface contract.

**Why separate agents?**

Separation allows: (a) different context for each domain (indicators for technical, documents for fundamental, news for sentiment); (b) independent confidence scoring per domain; (c) independent failure — if sentiment data is unavailable, fundamental and technical still complete; (d) the Critic can review each agent's output independently.

**How does the critic work?**

The Critic receives all three specialist reports as structured text and is instructed to find: unsupported claims (FACT with no evidence IDs), contradictions between agents (e.g., fundamental says margins improving, sentiment says pricing pressure), overconfident ratings on thin evidence, and stale or missing data. It returns a list of `CriticFinding` objects with severity ratings and recommended corrections.

**How do agents disagree?**

The Synthesis agent receives all reports including critic findings and is explicitly instructed to identify and report disagreements rather than resolve them artificially. The final `ResearchReport` has a `conflicting_signals` field listing topics where agents disagree, which agent holds which view, and the suggested resolution.

---

## BACKEND

**How does FastAPI handle requests?**

FastAPI uses Starlette's ASGI server (uvicorn). Each request is handled in an asyncio coroutine. I/O-bound operations (LLM, DB, Redis) yield the event loop while waiting, allowing other requests to be served concurrently without threads.

**How does async execution work?**

`asyncio.gather()` schedules multiple coroutines and runs them concurrently within the event loop. Blocking operations (yfinance, pandas) run in a thread pool via `run_in_executor` so they don't block the event loop.

**Where can race conditions occur?**

The main one: duplicate `POST /research` with the same idempotency_key. Two simultaneous requests both pass the Redis idempotency check (Redis check is not atomic with the DB write). Mitigation: the PostgreSQL `idempotency_key` column has a UNIQUE constraint. The second request gets an IntegrityError and returns the first request's ID.

**How does idempotency work?**

Client sends a unique `idempotency_key` with the request. We check Redis for `idempotency:{key}`. If found, return the existing `research_id`. If not, proceed, persist the request, then write the mapping to Redis with a 24h TTL. This guarantees that the same key always returns the same research result within 24 hours.

**How does rate limiting work?**

Redis sorted-set sliding window. Each API call adds a timestamp to `rl:{identifier}:{key}`. We remove entries older than the window (ZREMRANGEBYSCORE) and count remaining entries (ZCARD). If count > limit, return 429. All three operations run in a single Redis pipeline for atomicity.

---

## DATABASE

**Why PostgreSQL?**

Single deployment for relational state + vector search + full-text search. ACID guarantees. pgvector for embeddings, tsvector for BM25-like ranking.

**What indexes did you create?**

- `chunks.embedding`: IVFFlat cosine index (approximate nearest-neighbour, sub-linear search)
- `chunks.text_tsv`: GIN index (fast full-text query matching)
- `documents.(ticker, document_type)`: composite index for filtered document queries
- `research_requests.(ticker, status)`: composite index for reporting queries
- `research_requests.idempotency_key`: unique index (also serves as dedup constraint)

**How do you handle concurrent requests?**

Connection pooling (pool_size=10, max_overflow=20 per worker). Async sessions via asyncpg. Each FastAPI request gets its own session from the pool. Session is committed or rolled back in a `finally` block.

---

## SYSTEM DESIGN — Scaling to 100k requests/day

**Bottleneck analysis:**
At 100k research requests/day ≈ 1.2 req/second average. Assuming 2-minute pipeline: 144 concurrent pipelines at peak. Each pipeline makes ~8 LLM calls. That's ~1,150 LLM calls/minute — within OpenAI's tier 4 rate limits.

**What would you cache?**

- Research results for identical (ticker, time window) pairs — most obvious win
- Retrieved chunks for the same query (retrieval cache, 30-min TTL)
- Financial metrics for the same ticker (5-min TTL)
- Technical indicators for the same ticker (5-min TTL)

**What becomes the bottleneck?**

OpenAI API rate limits and per-token costs. At scale, we would batch-process off-peak, implement a job queue (Celery + SQS), and potentially use a smaller model for the less-complex agents.

**How would you queue jobs?**

Redis Queue or Celery with SQS. `POST /research` writes a job to the queue and returns immediately. Workers (separate processes) consume jobs, run pipelines, write results to DB. Web API reads results from DB/cache.

**How would you horizontally scale agents?**

Extract agents into a separate worker service. Dispatch agent tasks via message queue (e.g., SQS). Multiple worker replicas process agent tasks in parallel. The orchestrator coordinates by waiting for all task results before proceeding.

---

## RELIABILITY

**What happens if one agent fails?**

Each agent wraps its execution in `try/except` and returns a `failed=True` `AgentReport` rather than raising. The pipeline continues. The synthesis agent notes the missing report. The final report is marked `PARTIAL` status rather than `FAILED`. The `confidence.data_completeness` score is reduced proportionally.

**What happens if Redis fails?**

Rate limiting and idempotency fall back to pass-through (allow the request). Research results are served from PostgreSQL (cache miss). The pipeline itself does not depend on Redis for correctness — only for caching and deduplication.

**What happens if the LLM provider times out?**

The `OpenAIProvider` retries up to 3 times with exponential backoff (2s, 4s, 8s). If all retries fail, the provider raises. The calling agent catches this and returns a failed AgentReport.

**How do you retry safely?**

Retries are safe because LLM calls are idempotent — the same prompt returns a (statistically similar) valid response on retry. We use exponential backoff to respect rate limits. We never retry non-idempotent operations (database writes) on failure — those raise and let the caller decide.

---

## EVALUATION

**How do you know the system is better than a single agent?**

We implement a baseline: a single LLM call with all context concatenated. We compare both against the fixed evaluation dataset on: evidence coverage (how many relevant chunks are cited), citation accuracy, hallucination rate (FACT findings without evidence IDs), and structured output validity.

**How do you measure hallucination?**

Proxy metric: fraction of FACT-type findings that have no evidence_ids. A claim typed as FACT but unsupported by any retrieved chunk is a hallucination risk. True hallucination detection (verifying each claim against source text) requires human labelling and is done periodically as a manual audit.

**How do you evaluate citation correctness?**

We compare cited chunk_ids against a human-labelled set of relevant chunks for a query. Citation accuracy = |cited ∩ relevant| / |cited|. A high citation accuracy means the agent is citing the right passages; low accuracy means it is citing tangentially related chunks (a form of confabulation).

**How do you measure retrieval quality?**

Precision@k, Recall@k, NDCG@k against human-labelled relevant chunk sets. We compare across four retrieval configurations to quantify the marginal value of each pipeline component.
