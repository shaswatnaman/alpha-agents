"""
API request/response schemas (separate from domain models).

These are the contracts exposed to external callers.
Domain models are richer internal representations;
these schemas control what is serialised over the wire.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.domain.models import (
    AgentRole,
    ConfidenceScore,
    CriticFinding,
    Evidence,
    ResearchStatus,
    RiskFactor,
    Sentiment,
)


# ── Requests ──────────────────────────────────────────────────────────────────

class ResearchRequestBody(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL, MSFT)")
    company_name: str | None = Field(None, description="Optional human-readable company name")
    document_ids: list[str] = Field(
        default_factory=list,
        description="IDs of pre-ingested documents to include in retrieval",
    )
    idempotency_key: str | None = Field(
        None,
        description="Unique client key; duplicate requests with the same key return the same result",
    )


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    already_existed: bool
    message: str


# ── Responses ─────────────────────────────────────────────────────────────────

class ResearchCreatedResponse(BaseModel):
    research_id: str
    status: ResearchStatus
    message: str


class ResearchStatusResponse(BaseModel):
    research_id: str
    ticker: str
    status: ResearchStatus
    created_at: datetime
    updated_at: datetime
    report_id: str | None = None
    error_message: str | None = None
    pipeline_stage: str | None = None   # live status from Redis


class AgentSummary(BaseModel):
    agent: str
    status: str   # "completed" | "failed" | "unavailable"
    confidence: float | None = None
    summary: str | None = None
    execution_time_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    failure_reason: str | None = None


class EvidenceSummary(BaseModel):
    id: str
    document_id: str
    chunk_id: str
    source_filename: str
    document_type: str
    quote: str
    relevance_score: float
    retrieval_method: str


class ResearchReportResponse(BaseModel):
    research_id: str
    ticker: str
    company_name: str | None
    executive_summary: str
    fundamental_view: str
    technical_view: str
    sentiment_view: str
    key_risks: list[RiskFactor]
    critic_findings: list[CriticFinding]
    overall_sentiment: Sentiment
    confidence: ConfidenceScore
    citation_count: int
    created_at: datetime
    total_execution_time_ms: int
    total_tokens: int
    estimated_cost_usd: float


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
    details: Any = None
