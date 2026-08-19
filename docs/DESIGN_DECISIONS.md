# Design Decisions — AlphaAgents

---

## Why FastAPI?

**Problem:** We need an async HTTP framework that handles concurrent LLM API calls without blocking.

**Options:**
- Django: synchronous-first ORM; async support is bolted-on
- Flask: synchronous; async requires Quart fork
- FastAPI: async-native, Pydantic integration built-in, auto-generates OpenAPI docs

**Decision:** FastAPI. Async-first execution model matches the I/O-heavy LLM and database workload. Pydantic integration eliminates boilerplate serialisation. Background tasks (`asyncio.create_task`) for non-blocking research pipelines.

**Tradeoffs:** FastAPI lacks Django's batteries (admin, ORM, auth). We accepted this — we don't need an admin panel and we want to control our DB layer.

---

## Why PostgreSQL over a dedicated vector database?

**Problem:** We need both relational state (research requests, agent runs) and vector similarity search (chunk retrieval).

**Options:**
- PostgreSQL + pgvector: one deployment, ACID for all state, vector search built in
- Pinecone/Weaviate + PostgreSQL: two deployments, two connection pools, two failure modes

**Decision:** PostgreSQL + pgvector. At the document volumes expected (< 1M chunks per deployment), pgvector's IVFFlat index provides adequate recall and latency. Eliminating the operational burden of a second database is worth more than the marginal performance gain of a specialised vector store.

**Tradeoffs:** At 100M+ chunks, pgvector's performance would degrade and a dedicated ANN index (HNSW via Weaviate or Qdrant) would be necessary.

---

## Why Redis?

**Problem:** We have multiple distinct caching needs that a PostgreSQL-only approach would handle poorly.

**Redis use cases in this system:**
1. **Research result cache** (TTL=1h): Serves repeat reads of completed reports without hitting PostgreSQL
2. **Idempotency keys** (TTL=24h): Deduplicates concurrent duplicate submissions with sub-millisecond lookup
3. **Rate limiting** (sliding window sorted sets): Redis atomic sorted-set operations (`ZADD`, `ZREMRANGEBYSCORE`, `ZCARD`) in a single pipeline make sliding-window rate limiting correct under concurrency
4. **Live pipeline status** (TTL=1h): Real-time stage tracking without polling the DB

PostgreSQL could handle cases 1 and 2 via table reads, but at higher latency and write amplification. Cases 3 and 4 benefit specifically from Redis' atomic operations and key-expiry primitives.

**Tradeoffs:** Redis is an in-memory store. A restart without persistence loses all cached data. This is acceptable — cached research results are reconstructable from PostgreSQL; rate limit buckets reset on restart (a brief window of elevated limits is acceptable).

---

## Why hybrid retrieval (dense + lexical + RRF)?

**Problem:** Dense retrieval alone fails on exact-match queries; lexical retrieval alone fails on semantic queries.

**Decision:** Run both, fuse with RRF (no score normalisation needed), then rerank with MMR.

**Measured improvement over dense-only:** See `docs/EVALUATION.md`.

---

## Why MMR?

**Problem:** Without diversity enforcement, the top-k retrieved chunks are often near-identical paragraphs from the same annual report section. The LLM receives redundant context and may hallucinate information it "should have seen".

**Decision:** Maximum Marginal Relevance with λ=0.6 selects chunks that are both relevant to the query and diverse from each other.

**Tradeoffs:** MMR requires computing pairwise similarities between candidates, which is O(k × n) per query. At k=20 candidates and n=6 selected, this is 120 similarity computations — negligible.

---

## Why structured outputs over free-form text?

**Problem:** If the LLM returns free-form text, we need regex to extract the confidence score. Malformed output (e.g., confidence=1.5) silently produces wrong results.

**Decision:** Use OpenAI structured output mode (`beta.chat.completions.parse`) with Pydantic models as the response schema. The API guarantees schema-valid JSON, validated by Pydantic before any code uses it.

**Tradeoffs:** Structured output requires careful schema design. Very complex schemas slow down the LLM. We keep schemas to <20 fields per agent.

---

## Why deterministic technical indicators?

**Problem:** If we ask the LLM to "calculate RSI-14", it will produce plausible-sounding but wrong numbers. LLMs are not calculators.

**Decision:** All numerical indicators (RSI, MACD, SMA, volatility) are computed in Python using pandas/numpy from raw yfinance data. The LLM receives the pre-computed values as context and interprets them — it is never asked to calculate.

**Tradeoffs:** We are limited to indicators supported by yfinance's history data. Custom data sources would require building a separate data ingestion pipeline.

---

## Why a modular monolith over microservices?

**Problem:** Microservices add network hops, deployment complexity, and distributed tracing requirements for a system that runs at a modest scale.

**Decision:** One deployable unit with strictly-separated internal modules. Module boundaries are enforced by import rules (no circular imports between modules). Extraction to microservices is possible later if agent execution time or volume demands it.

**Tradeoffs:** Horizontal scaling requires scaling the entire application, not individual agents. A future architecture might extract the agent execution layer into a separate worker service behind a queue.

---

## Why evidence citations?

**Problem:** Without citations, users cannot verify whether the AI's claims are grounded in reality or hallucinated.

**Decision:** Every material `AgentFinding` with `claim_type=FACT` must include `evidence_ids` referencing specific `Evidence` objects. The final report exposes `citations` mapping each claim to its source chunks (document filename, quote, retrieval method).

**Tradeoffs:** Evidence-grounding makes agent prompts more complex. Agents must be explicitly instructed to use evidence IDs, and the critic must check whether they have done so.
