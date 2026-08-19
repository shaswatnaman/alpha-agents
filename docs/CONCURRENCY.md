# Concurrency Design — AlphaAgents

## Execution DAG

```
START
  │
  ├── fetch_technical_indicators()  ─┐
  ├── fetch_fundamental_metrics()   ─┤─ asyncio.gather() — concurrent
  └── fetch_news()                  ─┘
                │
  ├── rag_retrieval() — concurrent queries, sequential gather
                │
  ├── FundamentalAgent.run()   ─┐
  ├── TechnicalAgent.run()     ─┤─ asyncio.gather() — concurrent
  └── SentimentAgent.run()     ─┘
                │
  └── CriticAgent.run()    ── sequential (depends on all three above)
                │
  └── SynthesisAgent.run() ── sequential (depends on critic)
                │
  └── assemble_report(), persist_to_db, cache_in_redis
```

---

## Why Certain Stages Are Concurrent

### Stage 1: Data Fetchers

`fetch_technical_indicators`, `fetch_fundamental_metrics`, and `fetch_news` all make external I/O calls (yfinance, HTTP). They have no data dependency on each other. Running them concurrently reduces the wall-clock wait from 3× the slowest call to 1× the slowest call.

Each function runs in an executor thread (`asyncio.run_in_executor`) because yfinance uses blocking requests. This avoids blocking the event loop.

### Stage 2: Specialist Agents

The Fundamental, Technical, and Sentiment agents each receive different inputs:
- Fundamental: financial metrics + retrieved evidence
- Technical: pre-computed indicator values
- Sentiment: news articles

There is no data dependency between them. Running them concurrently via `asyncio.gather` reduces agent wait time from `T_f + T_t + T_s` to `max(T_f, T_t, T_s)`.

---

## Why Critic and Synthesis Run Sequentially

The Critic agent needs all three specialist reports to be complete before it can challenge them. Synthesis needs the Critic's findings. This is a genuine sequential dependency — there is no safe way to parallelize these.

---

## Concurrency Safety

**No shared mutable state during parallel agent execution.** Each agent:
1. Receives read-only inputs (domain model instances are frozen after creation)
2. Returns a new `AgentReport` without modifying any shared object
3. Makes its own independent LLM API calls

The database writes happen *after* the `asyncio.gather` returns, in a sequential loop. This is correct: even if two agents finish simultaneously, the DB writes are serialised by the event loop.

**Redis writes** during the pipeline (`research:status:*` keys) are independent per research_id and do not conflict.

---

## Potential Race Conditions

### Duplicate Request Race
Two simultaneous `POST /api/v1/research` calls with the same idempotency_key could both pass the Redis idempotency check before either writes its result.

**Mitigation:** The idempotency_key column in PostgreSQL has a `UNIQUE` constraint. The second write raises an `IntegrityError`, which is caught and treated as "already exists". The response returns the first request's ID.

### Agent Failure During Gather
If one agent in `asyncio.gather` raises instead of returning a failed report, the gather propagates the exception and other agents' work is lost.

**Mitigation:** Each agent wraps its entire execution in `try/except` and returns a failed `AgentReport` rather than raising. The outer pipeline never sees an exception from individual agents.

---

## Async vs Threading

We use `asyncio` (cooperative multitasking) for all I/O-bound work: LLM API calls, database queries, Redis operations. These release the event loop while waiting for network I/O.

We use thread pool executors (`run_in_executor`) for CPU-bound or blocking-I/O work: yfinance (blocking requests), pandas indicator computation. These avoid blocking the event loop.

This is the correct split: asyncio for async-native I/O, threads for blocking library calls.
