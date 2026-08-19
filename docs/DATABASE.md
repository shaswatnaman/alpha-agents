# Database Design — AlphaAgents

## Why PostgreSQL?

PostgreSQL handles three distinct concerns in one deployment:
1. **Relational state** — research requests, agent runs, audit trail
2. **Vector similarity search** — document chunk embeddings via `pgvector`
3. **Full-text search** — BM25-like ranking via `tsvector`

This eliminates the operational burden of a separate vector database (Pinecone, Weaviate, Chroma) while maintaining ACID guarantees for all application state.

---

## Schema

### `documents`
Stores ingested document metadata. The actual text is in `chunks`.

```
id            VARCHAR(36) PK
filename      VARCHAR(512)
content_hash  VARCHAR(64) UNIQUE   -- SHA-256; prevents re-ingestion
ticker        VARCHAR(20) INDEX
document_type VARCHAR(50)
source_url    TEXT
published_date TIMESTAMPTZ
fiscal_year   INTEGER
fiscal_quarter INTEGER
created_at    TIMESTAMPTZ
```

**Index rationale:** `(ticker, document_type)` composite index supports the common query "give me all annual reports for AAPL".

### `chunks`
Text chunks with vector embeddings for retrieval.

```
id            VARCHAR(36) PK
document_id   FK → documents.id (CASCADE DELETE)
chunk_index   INTEGER
text          TEXT
char_start    INTEGER
char_end      INTEGER
section_title VARCHAR(512)
page_number   INTEGER
embedding     vector(1536)        -- pgvector
text_tsv      tsvector            -- full-text index
chunk_metadata JSONB
created_at    TIMESTAMPTZ
```

**Index rationale:**
- `embedding` uses IVFFlat (approximate nearest-neighbour) with cosine ops for sub-linear search time
- `text_tsv` uses GIN index for fast full-text queries
- IVFFlat `lists=100` is appropriate for up to ~1M vectors; increase to 200 at 4M+ vectors

### `research_requests`
Every API call to `POST /api/v1/research`.

```
id                 VARCHAR(36) PK
ticker             VARCHAR(20) INDEX
company_name       VARCHAR(256)
idempotency_key    VARCHAR(256) UNIQUE INDEX
status             VARCHAR(20)   -- pending|running|completed|failed|partial
report_id          VARCHAR(36)
error_message      TEXT
requested_by       VARCHAR(256)
created_at         TIMESTAMPTZ INDEX
updated_at         TIMESTAMPTZ
```

**Index rationale:** `(ticker, status)` supports "show me all completed runs for AAPL". `created_at` supports time-based pagination.

### `agent_runs`
One row per agent per research pipeline execution. Enables debugging individual agent failures.

```
id                VARCHAR(36) PK
research_id       FK → research_requests.id
agent_role        VARCHAR(50)
status            VARCHAR(20)
output_json       JSONB        -- full AgentReport
confidence        FLOAT
execution_time_ms INTEGER
token_usage       JSONB
failure_reason    TEXT
created_at        TIMESTAMPTZ
```

### `research_reports`
Final assembled report. Also cached in Redis; this is the durable copy.

```
id                     VARCHAR(36) PK
research_id            VARCHAR(36) UNIQUE INDEX
ticker                 VARCHAR(20) INDEX
report_json            JSONB        -- full ResearchReport
total_tokens           INTEGER
estimated_cost_usd     FLOAT
total_execution_time_ms INTEGER
created_at             TIMESTAMPTZ INDEX
```

### `evaluation_results`
Stores experiment results from the evaluation runner.

---

## Connection Pool Configuration

```
pool_size=10       # 10 persistent connections per worker
max_overflow=20    # up to 20 additional connections on burst
```

With 2 uvicorn workers, peak DB connections = 2 × (10 + 20) = 60. Well within PostgreSQL's default `max_connections=100`.

---

## Handling Concurrent Writes

The `idempotency_key` UNIQUE constraint means simultaneous duplicate requests cause a `UniqueConstraintViolation` on the second write. The application catches this and returns the existing `research_id` — no phantom jobs are created.

---

## Data Retention

Document chunks and embeddings are retained indefinitely (they represent the client's uploaded documents). Research reports are retained indefinitely. The Redis cache expires after 1 hour (TTL) and serves as a fast read path for recent results.

For a production deployment, consider adding a `deleted_at` soft-delete column and a periodic cleanup job for old reports.
