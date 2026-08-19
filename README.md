# AlphaAgents

**Production-grade multi-agent financial research platform.**

AlphaAgents is an AI-assisted financial research system that takes a company ticker and produces a structured, evidence-grounded research report by orchestrating multiple specialised AI agents across fundamental analysis, technical analysis, and news sentiment — then synthesising their findings through an adversarial critic agent.

> **Disclaimer:** This is a research and decision-support tool. It does not constitute investment advice. No financial predictions are made.

---

## Overview

```
User / API Client
      │
      ▼
FastAPI Gateway (auth · rate limiting · idempotency)
      │
Research Orchestrator (async background task)
      │
      ├── [concurrent] FundamentalAgent  ← financial metrics + document RAG
      ├── [concurrent] TechnicalAgent    ← deterministic computed indicators
      └── [concurrent] SentimentAgent    ← real fetched news articles
                     │
               CriticAgent  ← challenges claims, flags contradictions
                     │
             SynthesisAgent  ← integrates all views into final report
                     │
          Structured ResearchReport
          (citations · confidence · conflicts)
```

---

## Why Multi-Agent?

| Concern | Multi-Agent Benefit |
|---|---|
| Domain isolation | Each agent receives only domain-appropriate context |
| Independent failure | If sentiment data is unavailable, fundamental analysis still completes |
| Attribution | Every claim is traceable to a specific agent and evidence chunk |
| Adversarial review | The Critic agent challenges unsupported claims across all agents |
| Concurrency | Fundamental, Technical, and Sentiment agents run in parallel |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a detailed explanation.

---

## RAG Pipeline

```
Query
  → DenseRetriever    (pgvector cosine similarity)
  → LexicalRetriever  (PostgreSQL full-text tsvector/BM25-like)
  → RRF Fusion        (Reciprocal Rank Fusion — no score normalisation needed)
  → MMR Reranking     (Maximum Marginal Relevance — diversity enforcement)
  → Evidence objects  (document_id · chunk_id · quote · source)
```

See [`docs/RAG.md`](docs/RAG.md) for the full explanation including failure modes and evaluation methodology.

---

## Agent Architecture

| Agent | Input | Output |
|---|---|---|
| Fundamental | Financial metrics + RAG evidence | Findings (FACT/INFERENCE/UNCERTAINTY) + risks + confidence |
| Technical | Pre-computed indicators (RSI, MACD, SMA) | Trend/momentum findings + sentiment |
| Sentiment | Fetched news articles | Sentiment signals + key events |
| Critic | All three specialist reports | Challenges + severity ratings + recommendations |
| Synthesis | All reports + critic findings | Executive summary + per-dimension views + conflicts |

**Every LLM output is schema-validated via Pydantic.** There is no regex parsing of free-form text.

---

## Evidence & Citations

Every material FACT claim in the final report references one or more `Evidence` objects:

```json
{
  "claim": "Revenue grew 12% year-over-year.",
  "evidence": [
    {
      "document_id": "...",
      "chunk_id": "...",
      "source_filename": "AAPL_10K_2024.pdf",
      "quote": "Net sales increased 12 percent compared to the prior year...",
      "relevance_score": 0.91,
      "retrieval_method": "mmr"
    }
  ]
}
```

The API exposes `GET /api/v1/research/{id}/evidence` to inspect all citations.

---

## API

```
POST   /api/v1/research                        Start a research pipeline
GET    /api/v1/research/{id}                   Poll status
GET    /api/v1/research/{id}/report            Get the final report
GET    /api/v1/research/{id}/agents            Get individual agent outputs
GET    /api/v1/research/{id}/evidence          Get all evidence citations
POST   /api/v1/documents                       Upload a document for RAG
GET    /api/v1/documents/{ticker}              List documents for a ticker
GET    /health                                 Health check
GET    /metrics                                Prometheus metrics
```

Authentication: `X-API-Key` header required. Configure keys in `VALID_API_KEYS` env var.

See [`docs/API.md`](docs/API.md) for full request/response schemas.

---

## Database

PostgreSQL + pgvector. One deployment for relational state and vector search.

Key tables: `documents` · `chunks` (with vector and tsvector indexes) · `research_requests` · `agent_runs` · `research_reports` · `evaluation_results`

See [`docs/DATABASE.md`](docs/DATABASE.md).

---

## Caching

| Cache | Backend | TTL | Purpose |
|---|---|---|---|
| Research reports | Redis | 1 hour | Avoid re-reading from PostgreSQL for repeat requests |
| Idempotency keys | Redis | 24 hours | Deduplicate concurrent duplicate submissions |
| Rate limit buckets | Redis | 60s / 1h | Sliding-window rate limiting per API key |
| Pipeline status | Redis | 1 hour | Real-time stage tracking |

---

## Reliability

- **Partial failure:** A failed agent returns a `failed=True` `AgentReport`. The pipeline continues with the remaining agents. The report is marked `PARTIAL`.
- **LLM retries:** Exponential backoff (3 retries, 2s/4s/8s) on `RateLimitError` and `APITimeoutError`.
- **Idempotency:** Duplicate requests with the same key return the existing result without re-running the pipeline.
- **Database protection:** `UNIQUE` constraint on `idempotency_key` prevents phantom jobs under race conditions.

---

## Observability

Prometheus metrics at `GET /metrics`:

| Metric | Description |
|---|---|
| `alphaagents_llm_tokens_total` | LLM token consumption by model and direction |
| `alphaagents_llm_cost_usd_total` | Estimated USD cost by model |
| `alphaagents_llm_latency_seconds` | LLM call latency histogram |
| `alphaagents_agent_execution_seconds` | Agent wall-clock time by agent and status |
| `alphaagents_agent_failures_total` | Agent failure counts by agent and reason |
| `alphaagents_retrieval_latency_seconds` | Retrieval latency by method |
| `alphaagents_cache_hits_total` | Cache hits by cache name |
| `alphaagents_research_total` | Research completions by status |
| `alphaagents_active_research` | Currently running pipelines |

Grafana dashboard available at `http://localhost:3000` (Docker Compose).

See [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md).

---

## Evaluation

```bash
# Run retrieval experiments (dense-only baseline vs. hybrid+MMR)
python -m app.evaluation.runner --experiment dense_only
python -m app.evaluation.runner --experiment hybrid_mmr
```

Metrics: Precision@6, Recall@6, NDCG@6 against a fixed human-labelled evaluation dataset.

See [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Testing

```bash
# Run unit tests (no external dependencies)
pytest tests/unit/ -v

# Run integration tests (mocked dependencies)
pytest tests/integration/ -v

# Run all tests with coverage
pytest --cov=app --cov-report=html
```

Test coverage includes: chunking, retrieval (MMR, RRF), evaluation metrics, domain model validation, technical indicator computation, agent failure handling, API authentication, API endpoint contracts.

---

## Docker Setup

```bash
# Copy env file and add your OpenAI key
cp .env.example .env

# Start all services
docker compose up -d

# Check health
curl http://localhost:8000/health
```

Services:
- `app` — FastAPI on :8000
- `postgres` — PostgreSQL + pgvector on :5432
- `redis` — Redis on :6379
- `prometheus` — on :9090
- `grafana` — on :3000 (admin/admin)

---

## Local Development

```bash
# Create virtualenv and install dependencies
pip install uv
uv sync --all-extras

# Set environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Start PostgreSQL and Redis (via Docker or local install)
docker compose up postgres redis -d

# Run database migrations
alembic upgrade head

# Start the API
uvicorn app.main:app --reload
```

---

## Example Research

```bash
# Start a research pipeline
curl -X POST http://localhost:8000/api/v1/research \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "idempotency_key": "my-unique-request-id"}'

# Poll for status
curl http://localhost:8000/api/v1/research/{research_id} \
  -H "X-API-Key: dev-key-1"

# Get the report when status=completed
curl http://localhost:8000/api/v1/research/{research_id}/report \
  -H "X-API-Key: dev-key-1"

# Inspect evidence citations
curl http://localhost:8000/api/v1/research/{research_id}/evidence \
  -H "X-API-Key: dev-key-1"
```

---

## Limitations

1. **LLM accuracy:** Structured output validation prevents malformed responses but cannot verify factual accuracy. The Critic agent reduces (not eliminates) hallucinations.
2. **News coverage:** Yahoo Finance news coverage varies by ticker. International or small-cap stocks may have sparse news.
3. **Document dependency:** Fundamental analysis quality depends on the quality of documents ingested via `POST /api/v1/documents`. Without relevant documents, the agent relies entirely on yfinance structured data.
4. **Technical analysis lag:** All indicators are computed from daily close prices. Intraday signals are not supported.
5. **No investment advice:** This is a research tool. It does not produce BUY/SELL recommendations.

---

## Future Improvements

- [ ] HNSW index for pgvector at scale (> 1M chunks)
- [ ] Job queue (Redis Queue / Celery) for decoupled pipeline execution
- [ ] WebSocket endpoint for real-time pipeline progress streaming
- [ ] SEC EDGAR integration for automatic document ingestion
- [ ] Cross-ticker peer comparison agent
- [ ] Earnings calendar integration
- [ ] OpenTelemetry distributed tracing

---

## Architecture Documents

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture and module map
- [`docs/RAG.md`](docs/RAG.md) — RAG pipeline design and theory
- [`docs/AGENTS.md`](docs/AGENTS.md) — Agent responsibilities and prompts
- [`docs/DATABASE.md`](docs/DATABASE.md) — Schema, indexes, connection pooling
- [`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) — Async execution and race conditions
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — Evaluation framework and methodology
- [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) — Metrics, logging, tracing
- [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) — Why each technology was chosen
- [`docs/INTERVIEW.md`](docs/INTERVIEW.md) — Engineering interview Q&A

---

## License

MIT
