# Observability — AlphaAgents

## Structured Logging

Every log record includes: `timestamp`, `level`, `logger`, `request_id` (when in a request context), and all caller-provided key-value pairs. Output format is JSON in production, coloured human-readable in development.

Key log events:

| Event | Level | Fields |
|---|---|---|
| `http_request` | INFO | method, path, status_code, latency_ms |
| `llm_complete` | INFO | model, input_tokens, output_tokens, latency_ms, cost_usd |
| `llm_retry` | WARNING | attempt, backoff_seconds, model |
| `pipeline_step` | INFO | step, research_id |
| `pipeline_complete` | INFO | research_id, ticker, elapsed_ms, total_tokens, status |
| `pipeline_error` | ERROR | research_id, error (with traceback) |
| `retrieval_complete` | INFO | query_preview, ticker, dense_count, lexical_count, final_count, latency_ms |
| `rate_limit_exceeded` | WARNING | key, identifier, count, limit |
| `idempotency_hit` | INFO | key |

## Prometheus Metrics

Scraped at `GET /metrics`. Configured in `monitoring/prometheus.yml`.

### LLM Metrics
- `alphaagents_llm_tokens_total{model, direction}` — counter
- `alphaagents_llm_cost_usd_total{model}` — counter (estimated)
- `alphaagents_llm_latency_seconds{model}` — histogram (p50, p95, p99)

### Agent Metrics
- `alphaagents_agent_execution_seconds{agent, status}` — histogram
- `alphaagents_agent_failures_total{agent, reason}` — counter

### Retrieval Metrics
- `alphaagents_retrieval_latency_seconds{method}` — histogram per retrieval method
- `alphaagents_retrieval_chunks_total{method}` — counter

### Cache Metrics
- `alphaagents_cache_hits_total{cache_name}` — counter
- `alphaagents_cache_misses_total{cache_name}` — counter

### HTTP Metrics
- `alphaagents_http_requests_total{method, path, status_code}` — counter
- `alphaagents_http_latency_seconds{method, path}` — histogram

### Business Metrics
- `alphaagents_research_total{status}` — counter (completed/failed/partial)
- `alphaagents_research_latency_seconds` — histogram (end-to-end pipeline)
- `alphaagents_active_research` — gauge

## Grafana

Dashboard available at `http://localhost:3000` (admin/admin) after `docker compose up`.

Panels: LLM cost/hour, agent failure rate, cache hit rate, research latency p95, active pipelines.

## Key Alerts (to configure in Grafana)

- `agent_failure_rate > 10%` over 5 minutes → PagerDuty
- `llm_latency_p95 > 30s` → warning
- `active_research > 50` → capacity warning
- `cache_hit_rate < 50%` for `research_report` → investigate
