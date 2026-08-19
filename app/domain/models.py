"""
Core domain models — pure Pydantic, no I/O.

These are the lingua franca of the entire system.  Every agent, service,
and repository speaks in terms of these types.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Identifiers ───────────────────────────────────────────────────────────────


def new_id() -> str:
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────


class ResearchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # completed with some agent failures


class AgentRole(StrEnum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    CRITIC = "critic"
    SYNTHESIS = "synthesis"


class Sentiment(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ClaimType(StrEnum):
    FACT = "fact"  # directly supported by evidence
    INFERENCE = "inference"  # derived from facts via reasoning
    UNCERTAINTY = "uncertainty"  # explicitly flagged as uncertain


class DocumentType(StrEnum):
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    SEC_FILING = "sec_filing"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    NEWS_ARTICLE = "news_article"
    ANALYST_REPORT = "analyst_report"
    OTHER = "other"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Base ──────────────────────────────────────────────────────────────────────


class DomainModel(BaseModel):
    model_config = ConfigDict(
        frozen=False,
        use_enum_values=True,
        populate_by_name=True,
    )


# ── Documents & Chunks ────────────────────────────────────────────────────────


class DocumentMetadata(DomainModel):
    ticker: str
    company_name: str | None = None
    document_type: DocumentType = DocumentType.OTHER
    source_url: str | None = None
    published_date: datetime | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    page_count: int | None = None


class Document(DomainModel):
    id: str = Field(default_factory=new_id)
    filename: str
    content_hash: str  # SHA-256 of raw bytes; prevents re-ingestion
    metadata: DocumentMetadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(DomainModel):
    id: str = Field(default_factory=new_id)
    document_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    section_title: str | None = None
    page_number: int | None = None
    embedding: list[float] | None = None  # populated after embedding step
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Evidence ──────────────────────────────────────────────────────────────────


class Evidence(DomainModel):
    """A traceable piece of retrieved context backing an agent claim."""

    id: str = Field(default_factory=new_id)
    document_id: str
    chunk_id: str
    source_filename: str
    document_type: DocumentType
    published_date: datetime | None = None
    quote: str  # verbatim excerpt from the chunk
    relevance_score: float  # similarity score from retrieval (0-1)
    retrieval_method: str  # "dense" | "lexical" | "hybrid"


# ── Agent Outputs ─────────────────────────────────────────────────────────────


class AgentFinding(DomainModel):
    """A single factual or inferred finding from an agent."""

    claim: str
    claim_type: ClaimType
    evidence_ids: list[str] = Field(default_factory=list)  # references Evidence.id
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class RiskFactor(DomainModel):
    description: str
    severity: Severity
    evidence_ids: list[str] = Field(default_factory=list)
    mitigation: str | None = None


class CriticFinding(DomainModel):
    """Output of the Critic/Risk agent challenging another agent's output."""

    affected_agent: AgentRole
    affected_claim: str
    issue: str
    severity: Severity
    recommendation: str


class TechnicalIndicators(DomainModel):
    """Deterministically computed market indicators — never hallucinated."""

    ticker: str
    as_of_date: datetime
    current_price: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    annualized_volatility: float | None = None
    annualized_return_3m: float | None = None
    volume_avg_20d: float | None = None
    price_52w_high: float | None = None
    price_52w_low: float | None = None
    computation_errors: list[str] = Field(default_factory=list)


class FundamentalMetrics(DomainModel):
    """Key financial metrics fetched from structured data sources."""

    ticker: str
    as_of_date: datetime
    market_cap: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    revenue_ttm: float | None = None
    revenue_growth_yoy: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    free_cash_flow: float | None = None
    eps_ttm: float | None = None
    dividend_yield: float | None = None
    fetch_errors: list[str] = Field(default_factory=list)


class AgentReport(DomainModel):
    """Structured output from a specialist agent."""

    agent: AgentRole
    research_id: str
    findings: list[AgentFinding]
    risks: list[RiskFactor] = Field(default_factory=list)
    sentiment: Sentiment | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: list[str] = Field(default_factory=list)
    summary: str
    execution_time_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    failed: bool = False
    failure_reason: str | None = None


# ── Research Report ───────────────────────────────────────────────────────────


class ConfidenceScore(DomainModel):
    overall: Annotated[float, Field(ge=0.0, le=1.0)]
    fundamental: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    technical: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    sentiment: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    data_completeness: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class Citation(DomainModel):
    claim: str
    evidence: list[Evidence]


class ConflictingSignal(DomainModel):
    topic: str
    agent_a: AgentRole
    claim_a: str
    agent_b: AgentRole
    claim_b: str
    resolution: str | None = None


class ResearchReport(DomainModel):
    """Final structured output of the full research pipeline."""

    id: str = Field(default_factory=new_id)
    research_id: str
    ticker: str
    company_name: str | None = None
    executive_summary: str
    fundamental_view: str
    technical_view: str
    sentiment_view: str
    key_risks: list[RiskFactor]
    conflicting_signals: list[ConflictingSignal]
    citations: list[Citation]
    confidence: ConfidenceScore
    overall_sentiment: Sentiment
    critic_findings: list[CriticFinding]
    agent_reports: dict[str, AgentReport]  # keyed by AgentRole value
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_execution_time_ms: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


# ── Research Request ──────────────────────────────────────────────────────────


class ResearchRequest(DomainModel):
    id: str = Field(default_factory=new_id)
    ticker: str
    company_name: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    status: ResearchStatus = ResearchStatus.PENDING
    report_id: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    requested_by: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()


# ── Evaluation ────────────────────────────────────────────────────────────────


class RetrievalEvalSample(DomainModel):
    query: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    ndcg: float


class EvaluationResult(DomainModel):
    id: str = Field(default_factory=new_id)
    experiment_name: str
    research_id: str | None = None
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    retrieval_ndcg: float | None = None
    citation_accuracy: float | None = None
    hallucination_rate: float | None = None
    agent_agreement_rate: float | None = None
    structured_output_validity: float | None = None
    latency_ms: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
