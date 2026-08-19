# AlphaAgents — Architecture

## System Overview

AlphaAgents is a **modular monolith** — a single deployable unit partitioned into strongly-separated internal modules, each with its own responsibility boundary and interface. This avoids the operational complexity of microservices while preserving the ability to extract modules into services later.

```
User / API Client
      │
      ▼
┌─────────────────────────────────────────────────┐
│                FastAPI Gateway                  │
│  • Auth middleware (API key)                    │
│  • Rate limiting (Redis sliding window)         │
│  • Request ID injection                         │
│  • Structured error handling                    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           Research Orchestrator                 │
│  • Idempotency check (Redis + PostgreSQL)       │
│  • Job creation & status tracking              │
│  • Concurrent agent dispatch (asyncio)          │
│  • Partial-failure degradation                  │
└──┬──────────────┬─────────────────┬─────────────┘
   │              │                 │
   ▼              ▼                 ▼
Fundamental    Technical        Sentiment
  Agent          Agent            Agent
   │              │                 │
   └──────────────┴─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Evidence / RAG  │
        │  Layer           │
        │  • Dense retrieval│
        │  • Lexical BM25  │
        │  • Hybrid merge  │
        │  • MMR rerank    │
        └────────┬─────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
   PostgreSQL         pgvector
   (relational        (vector
    state)             search)
                 │
                 ▼
           Critic Agent
                 │
                 ▼
         Synthesis Agent
                 │
                 ▼
    Structured Research Report
                 │
           ┌─────┴─────┐
           │           │
           ▼           ▼
         REST       WebSocket /
         API        SSE Stream
```

---

## Module Map

```
app/
├── api/              # HTTP layer: routes, middleware, exception handlers
│   └── v1/           # Versioned endpoints
├── agents/           # Specialist AI agents (each in its own file)
├── domain/           # Pure Pydantic domain models — no I/O allowed here
├── services/         # Orchestration, business logic, use-case coordinators
├── retrieval/        # RAG pipeline (dense, lexical, hybrid, reranking)
├── ingestion/        # Document ingestion pipeline (extract → chunk → embed → index)
├── llm/              # LLM provider abstraction and OpenAI implementation
├── data/             # Market data fetchers, technical indicator calculators
├── repositories/     # All database access (async SQLAlchemy)
├── cache/            # Redis client, caching decorators, rate limiter
├── evaluation/       # Evaluation framework, metric computation, baseline runner
├── observability/    # Structured logging, Prometheus metrics, trace context
├── config/           # Pydantic Settings, environment validation
└── schemas/          # API request/response schemas (different from domain models)
```

---

## Key Design Choices

### Modular Monolith over Microservices
Each internal module has a defined boundary (clear imports, no circular deps). A future extraction to microservices is feasible — but the operational overhead of distributed systems is not justified at this scale.

### PostgreSQL + pgvector
A single PostgreSQL instance handles both relational state (research requests, agent runs, users) and vector similarity search (document embeddings). This eliminates the operational burden of a separate vector database while maintaining ACID guarantees for all state.

### Concurrent Agent Execution
Fundamental, Technical, and Sentiment agents have no data dependency on each other. They run concurrently via `asyncio.gather`. Critic and Synthesis agents run sequentially after all three complete.

### Evidence-First Design
The domain model treats `Evidence` as a first-class citizen. Every `AgentFinding` must reference one or more `Evidence` objects, each of which carries a `document_id`, `chunk_id`, and `quote`. The Synthesis agent cannot make a claim without tracing it to evidence.

### Provider-Agnostic LLM Interface
The `LLMProvider` protocol defines a single `complete()` method. The `OpenAIProvider` implementation handles retries, timeouts, token counting, and cost estimation. Swapping to Anthropic or a local model requires implementing one protocol class.

### Pydantic-First Structured Outputs
Every agent uses OpenAI's structured output mode to return a Pydantic model. There is no regex parsing of free-form text. Invalid outputs raise `OutputValidationError` and trigger the retry/degradation path.
