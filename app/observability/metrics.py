"""
Prometheus metrics definitions.

Import this module early so metrics are registered before any code uses them.
All metric labels are kept intentionally narrow to avoid high-cardinality explosions.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── LLM ──────────────────────────────────────────────────────────────────────

llm_token_counter = Counter(
    "alphaagents_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "direction"],  # direction: input | output
)

llm_cost_counter = Counter(
    "alphaagents_llm_cost_usd_total",
    "Estimated LLM cost in USD",
    ["model"],
)

llm_latency_histogram = Histogram(
    "alphaagents_llm_latency_seconds",
    "LLM request latency",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# ── Agents ────────────────────────────────────────────────────────────────────

agent_execution_histogram = Histogram(
    "alphaagents_agent_execution_seconds",
    "Agent execution wall-clock time",
    ["agent", "status"],  # status: success | failure
    buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0],
)

agent_failure_counter = Counter(
    "alphaagents_agent_failures_total",
    "Agent execution failures",
    ["agent", "reason"],
)

# ── Retrieval ─────────────────────────────────────────────────────────────────

retrieval_latency_histogram = Histogram(
    "alphaagents_retrieval_latency_seconds",
    "RAG retrieval latency",
    ["method"],  # dense | lexical | hybrid | mmr
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

retrieval_chunk_counter = Counter(
    "alphaagents_retrieval_chunks_total",
    "Total chunks retrieved",
    ["method"],
)

# ── Cache ─────────────────────────────────────────────────────────────────────

cache_hit_counter = Counter(
    "alphaagents_cache_hits_total",
    "Cache hits",
    ["cache_name"],
)

cache_miss_counter = Counter(
    "alphaagents_cache_misses_total",
    "Cache misses",
    ["cache_name"],
)

# ── HTTP ──────────────────────────────────────────────────────────────────────

http_request_counter = Counter(
    "alphaagents_http_requests_total",
    "HTTP requests received",
    ["method", "path", "status_code"],
)

http_latency_histogram = Histogram(
    "alphaagents_http_latency_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Research ──────────────────────────────────────────────────────────────────

research_counter = Counter(
    "alphaagents_research_total",
    "Research requests processed",
    ["status"],  # completed | failed | partial
)

research_latency_histogram = Histogram(
    "alphaagents_research_latency_seconds",
    "End-to-end research pipeline latency",
    buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

active_research_gauge = Gauge(
    "alphaagents_active_research",
    "Currently running research pipelines",
)
