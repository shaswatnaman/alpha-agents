"""
Research endpoints — the primary API surface.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.rate_limiter import api_limiter, research_limiter
from app.cache.redis_client import get_redis
from app.config.settings import get_settings
from app.domain.models import ResearchRequest, ResearchStatus
from app.repositories.database import get_db
from app.repositories.research_repository import ResearchRepository
from app.schemas.api import (
    AgentSummary,
    EvidenceSummary,
    ResearchCreatedResponse,
    ResearchReportResponse,
    ResearchRequestBody,
    ResearchStatusResponse,
)
from app.services.orchestrator import ResearchOrchestrator

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/research", tags=["Research"])


def _get_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    settings = get_settings()
    if x_api_key not in settings.api_keys_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_api_key", "message": "A valid X-API-Key header is required"},
        )
    return x_api_key


@router.post(
    "",
    response_model=ResearchCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new research pipeline for a ticker",
)
async def create_research(
    body: ResearchRequestBody,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(_get_api_key),
) -> ResearchCreatedResponse:
    # Rate limiting: per API key
    await api_limiter.check(api_key)
    await research_limiter.check(api_key)

    request = ResearchRequest(
        ticker=body.ticker,
        company_name=body.company_name,
        document_ids=body.document_ids,
        idempotency_key=body.idempotency_key,
        requested_by=api_key,
    )

    repo = ResearchRepository(db)
    orchestrator = ResearchOrchestrator(repo)
    research_id = await orchestrator.start_research(request)

    return ResearchCreatedResponse(
        research_id=research_id,
        status=ResearchStatus.PENDING,
        message=f"Research pipeline started for {body.ticker}. Poll GET /research/{research_id} for status.",
    )


@router.get(
    "/{research_id}",
    response_model=ResearchStatusResponse,
    summary="Get the status of a research request",
)
async def get_research_status(
    research_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(_get_api_key),
) -> ResearchStatusResponse:
    await api_limiter.check(api_key)

    repo = ResearchRepository(db)
    req = await repo.get_request(research_id)
    if req is None:
        raise HTTPException(
            status_code=404, detail={"error": "not_found", "message": "Research request not found"}
        )

    # Live pipeline stage from Redis
    redis = await get_redis()
    stage = await redis.get(f"research:status:{research_id}")

    return ResearchStatusResponse(
        research_id=req.id,
        ticker=req.ticker,
        status=req.status,
        created_at=req.created_at,
        updated_at=req.updated_at,
        report_id=req.report_id,
        error_message=req.error_message,
        pipeline_stage=stage,
    )


@router.get(
    "/{research_id}/report",
    response_model=ResearchReportResponse,
    summary="Get the final research report",
)
async def get_research_report(
    research_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(_get_api_key),
) -> ResearchReportResponse:
    await api_limiter.check(api_key)

    redis = await get_redis()

    # Try Redis cache first
    from app.observability.metrics import cache_hit_counter, cache_miss_counter

    cached = await redis.get(f"research:report:{research_id}")
    if cached:
        cache_hit_counter.labels(cache_name="research_report").inc()
        from app.domain.models import ResearchReport

        report = ResearchReport.model_validate_json(cached)
    else:
        cache_miss_counter.labels(cache_name="research_report").inc()
        repo = ResearchRepository(db)
        report = await repo.get_report(research_id)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "not_found",
                    "message": "Report not found or pipeline still running",
                },
            )

    return ResearchReportResponse(
        research_id=report.research_id,
        ticker=report.ticker,
        company_name=report.company_name,
        executive_summary=report.executive_summary,
        fundamental_view=report.fundamental_view,
        technical_view=report.technical_view,
        sentiment_view=report.sentiment_view,
        key_risks=report.key_risks,
        critic_findings=report.critic_findings,
        overall_sentiment=report.overall_sentiment,
        confidence=report.confidence,
        citation_count=len(report.citations),
        created_at=report.created_at,
        total_execution_time_ms=report.total_execution_time_ms,
        total_tokens=report.total_tokens,
        estimated_cost_usd=report.estimated_cost_usd,
    )


@router.get(
    "/{research_id}/agents",
    response_model=list[AgentSummary],
    summary="Get individual agent outputs for a research run",
)
async def get_agent_summaries(
    research_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(_get_api_key),
) -> list[AgentSummary]:
    await api_limiter.check(api_key)

    redis = await get_redis()
    cached = await redis.get(f"research:report:{research_id}")
    if not cached:
        repo = ResearchRepository(db)
        report = await repo.get_report(research_id)
        if report is None:
            raise HTTPException(
                status_code=404, detail={"error": "not_found", "message": "Report not found"}
            )
    else:
        from app.domain.models import ResearchReport

        report = ResearchReport.model_validate_json(cached)

    summaries = []
    for role, agent_report in report.agent_reports.items():
        summaries.append(
            AgentSummary(
                agent=role,
                status="failed" if agent_report.failed else "completed",
                confidence=agent_report.confidence,
                summary=agent_report.summary,
                execution_time_ms=agent_report.execution_time_ms,
                token_usage=agent_report.token_usage,
                failure_reason=agent_report.failure_reason,
            )
        )
    return summaries


@router.get(
    "/{research_id}/evidence",
    response_model=list[EvidenceSummary],
    summary="Get evidence citations for a research report",
)
async def get_evidence(
    research_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(_get_api_key),
) -> list[EvidenceSummary]:
    await api_limiter.check(api_key)

    redis = await get_redis()
    cached = await redis.get(f"research:report:{research_id}")
    if cached:
        from app.domain.models import ResearchReport

        report = ResearchReport.model_validate_json(cached)
    else:
        repo = ResearchRepository(db)
        report = await repo.get_report(research_id)
        if report is None:
            raise HTTPException(
                status_code=404, detail={"error": "not_found", "message": "Report not found"}
            )

    seen: set[str] = set()
    evidence_list = []
    for citation in report.citations:
        for ev in citation.evidence:
            if ev.id not in seen:
                seen.add(ev.id)
                evidence_list.append(
                    EvidenceSummary(
                        id=ev.id,
                        document_id=ev.document_id,
                        chunk_id=ev.chunk_id,
                        source_filename=ev.source_filename,
                        document_type=ev.document_type,
                        quote=ev.quote,
                        relevance_score=ev.relevance_score,
                        retrieval_method=ev.retrieval_method,
                    )
                )
    return evidence_list
