# API Reference — AlphaAgents

## Authentication

All endpoints require `X-API-Key` header. Valid keys are configured via `VALID_API_KEYS` env var (comma-separated).

```
X-API-Key: your-api-key
```

## Rate Limits

- General API: 20 requests/minute per API key
- Research creation: 10 requests/hour per API key

Exceeded limits return HTTP 429 with `Retry-After` header.

---

## Endpoints

### POST /api/v1/research

Start a research pipeline.

**Request:**
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",        // optional
  "document_ids": ["doc-uuid-1"],      // optional: pre-ingested documents
  "idempotency_key": "my-unique-key"   // optional: deduplication
}
```

**Response 202:**
```json
{
  "research_id": "uuid",
  "status": "pending",
  "message": "Research pipeline started..."
}
```

---

### GET /api/v1/research/{research_id}

Poll pipeline status.

**Response 200:**
```json
{
  "research_id": "uuid",
  "ticker": "AAPL",
  "status": "running",                // pending|running|completed|failed|partial
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:30",
  "report_id": null,                  // set when status=completed
  "error_message": null,
  "pipeline_stage": "running:agents"  // live stage from Redis
}
```

---

### GET /api/v1/research/{research_id}/report

Get the final research report. Returns 404 while pipeline is still running.

**Response 200:**
```json
{
  "research_id": "uuid",
  "ticker": "AAPL",
  "executive_summary": "...",
  "fundamental_view": "...",
  "technical_view": "...",
  "sentiment_view": "...",
  "key_risks": [
    {"description": "...", "severity": "medium", "evidence_ids": []}
  ],
  "critic_findings": [
    {
      "affected_agent": "fundamental",
      "affected_claim": "...",
      "issue": "...",
      "severity": "low",
      "recommendation": "..."
    }
  ],
  "overall_sentiment": "bullish",
  "confidence": {
    "overall": 0.72,
    "fundamental": 0.81,
    "technical": 0.65,
    "sentiment": 0.70,
    "data_completeness": 1.0
  },
  "citation_count": 14,
  "created_at": "2024-01-01T00:02:45",
  "total_execution_time_ms": 45230,
  "total_tokens": 12450,
  "estimated_cost_usd": 0.0032
}
```

---

### GET /api/v1/research/{research_id}/agents

Individual agent outputs with execution metadata.

**Response 200:**
```json
[
  {
    "agent": "fundamental",
    "status": "completed",
    "confidence": 0.81,
    "summary": "Strong revenue growth...",
    "execution_time_ms": 8230,
    "token_usage": {"input": 2340, "output": 580},
    "failure_reason": null
  }
]
```

---

### GET /api/v1/research/{research_id}/evidence

All evidence citations for a report.

**Response 200:**
```json
[
  {
    "id": "uuid",
    "document_id": "uuid",
    "chunk_id": "uuid",
    "source_filename": "AAPL_10K_2024.pdf",
    "document_type": "annual_report",
    "quote": "Net sales increased 12 percent...",
    "relevance_score": 0.91,
    "retrieval_method": "mmr"
  }
]
```

---

### POST /api/v1/documents

Upload a document for RAG. Multipart form.

**Fields:**
- `file` — PDF, TXT, or DOCX
- `ticker` — e.g. "AAPL"
- `document_type` — optional: annual_report, quarterly_report, etc.
- `fiscal_year` — optional integer
- `fiscal_quarter` — optional integer

**Response 201:**
```json
{
  "document_id": "uuid",
  "filename": "AAPL_10K_2024.pdf",
  "chunk_count": 142,
  "already_existed": false,
  "message": "Successfully ingested 142 chunks."
}
```

---

### GET /health

```json
{"status": "ok", "version": "1.0.0", "environment": "development"}
```

---

## Error Responses

All errors return `application/json`:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Limit: 20 per 60s",
  "request_id": "uuid",
  "details": null
}
```

Common error codes: `invalid_api_key`, `rate_limit_exceeded`, `not_found`, `unsupported_file_type`, `file_too_large`, `internal_server_error`.
