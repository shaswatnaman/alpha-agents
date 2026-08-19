from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import AgentReport, AgentRole, ResearchReport, ResearchRequest, ResearchStatus
from app.repositories.models import AgentRunORM, ResearchReportORM, ResearchRequestORM


class ResearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Research Requests ────────────────────────────────────────────────────

    async def create_request(self, req: ResearchRequest) -> None:
        orm = ResearchRequestORM(
            id=req.id,
            ticker=req.ticker,
            company_name=req.company_name,
            idempotency_key=req.idempotency_key,
            status=req.status,
            requested_by=req.requested_by,
        )
        self._session.add(orm)

    async def get_request(self, research_id: str) -> ResearchRequest | None:
        result = await self._session.execute(
            select(ResearchRequestORM).where(ResearchRequestORM.id == research_id)
        )
        row = result.scalar_one_or_none()
        return _request_orm_to_domain(row) if row else None

    async def update_request_status(
        self,
        research_id: str,
        status: ResearchStatus,
        report_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        values: dict = {"status": status.value, "updated_at": datetime.utcnow()}
        if report_id is not None:
            values["report_id"] = report_id
        if error_message is not None:
            values["error_message"] = error_message
        await self._session.execute(
            update(ResearchRequestORM)
            .where(ResearchRequestORM.id == research_id)
            .values(**values)
        )

    # ── Agent Runs ───────────────────────────────────────────────────────────

    async def save_agent_run(self, report: AgentReport) -> None:
        orm = AgentRunORM(
            research_id=report.research_id,
            agent_role=report.agent.value if hasattr(report.agent, "value") else report.agent,
            status="failed" if report.failed else "completed",
            output_json=json.loads(report.model_dump_json()),
            confidence=report.confidence,
            execution_time_ms=report.execution_time_ms,
            token_usage=report.token_usage,
            failure_reason=report.failure_reason,
        )
        self._session.add(orm)

    # ── Research Reports ─────────────────────────────────────────────────────

    async def save_report(self, report: ResearchReport) -> None:
        orm = ResearchReportORM(
            id=report.id,
            research_id=report.research_id,
            ticker=report.ticker,
            report_json=json.loads(report.model_dump_json()),
            total_tokens=report.total_tokens,
            estimated_cost_usd=report.estimated_cost_usd,
            total_execution_time_ms=report.total_execution_time_ms,
        )
        self._session.add(orm)

    async def get_report(self, research_id: str) -> ResearchReport | None:
        result = await self._session.execute(
            select(ResearchReportORM).where(ResearchReportORM.research_id == research_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return ResearchReport(**row.report_json)

    async def list_reports_by_ticker(self, ticker: str, limit: int = 10) -> list[ResearchReport]:
        result = await self._session.execute(
            select(ResearchReportORM)
            .where(ResearchReportORM.ticker == ticker)
            .order_by(ResearchReportORM.created_at.desc())
            .limit(limit)
        )
        return [ResearchReport(**row.report_json) for row in result.scalars().all()]


def _request_orm_to_domain(row: ResearchRequestORM) -> ResearchRequest:
    return ResearchRequest(
        id=row.id,
        ticker=row.ticker,
        company_name=row.company_name,
        idempotency_key=row.idempotency_key,
        status=ResearchStatus(row.status),
        report_id=row.report_id,
        error_message=row.error_message,
        requested_by=row.requested_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
