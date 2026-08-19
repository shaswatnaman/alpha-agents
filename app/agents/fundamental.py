"""
Fundamental Analyst Agent

Input:
- Pre-fetched financial metrics (FundamentalMetrics)
- Evidence retrieved from financial documents (list[Evidence])

Output:
- Structured AgentReport with findings referencing evidence IDs
"""

from __future__ import annotations

import time
from typing import Annotated

import structlog
from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.domain.models import (
    AgentFinding,
    AgentReport,
    AgentRole,
    Evidence,
    FundamentalMetrics,
    RiskFactor,
)
from app.observability.metrics import agent_execution_histogram

log = structlog.get_logger(__name__)


class FundamentalAgentOutput(BaseModel):
    """Structured LLM output schema for the fundamental agent."""

    findings: list[AgentFinding]
    risks: list[RiskFactor]
    summary: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


def _format_metrics(metrics: FundamentalMetrics) -> str:
    def fmt(v: float | None, suffix: str = "") -> str:
        if v is None:
            return "N/A"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B{suffix}"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M{suffix}"
        return f"{v:.4f}{suffix}"

    return f"""
FINANCIAL METRICS (as of {metrics.as_of_date.date()}):
- Market Cap: {fmt(metrics.market_cap)}
- Revenue (TTM): {fmt(metrics.revenue_ttm)}
- Revenue Growth (YoY): {fmt(metrics.revenue_growth_yoy, "%") if metrics.revenue_growth_yoy else "N/A"}
- Gross Margin: {f"{metrics.gross_margin * 100:.1f}%" if metrics.gross_margin else "N/A"}
- Operating Margin: {f"{metrics.operating_margin * 100:.1f}%" if metrics.operating_margin else "N/A"}
- Net Margin: {f"{metrics.net_margin * 100:.1f}%" if metrics.net_margin else "N/A"}
- Free Cash Flow: {fmt(metrics.free_cash_flow)}
- Debt/Equity: {fmt(metrics.debt_to_equity)}
- Current Ratio: {fmt(metrics.current_ratio)}
- P/E Ratio: {fmt(metrics.pe_ratio)}
- P/B Ratio: {fmt(metrics.pb_ratio)}
- EPS (TTM): {fmt(metrics.eps_ttm)}
- Dividend Yield: {f"{metrics.dividend_yield * 100:.2f}%" if metrics.dividend_yield else "N/A"}
Data errors: {metrics.fetch_errors or "None"}
""".strip()


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "No document evidence available."
    lines = []
    for _i, ev in enumerate(evidence, 1):
        date_str = ev.published_date.strftime("%Y-%m") if ev.published_date else "unknown date"
        lines.append(
            f"[{ev.id[:8]}] ({ev.source_filename}, {date_str}, {ev.retrieval_method}):\n"
            f'  "{ev.quote[:300]}"'
        )
    return "\n\n".join(lines)


class FundamentalAgent(BaseAgent):
    role = AgentRole.FUNDAMENTAL
    prompt_file = "fundamental_analyst.txt"

    async def run(
        self,
        research_id: str,
        ticker: str,
        metrics: FundamentalMetrics,
        evidence: list[Evidence],
    ) -> AgentReport:
        t0 = time.monotonic()

        user_message = f"""
COMPANY: {ticker}

{_format_metrics(metrics)}

RETRIEVED DOCUMENT EVIDENCE:
{_format_evidence(evidence)}

Based on the above, produce a structured fundamental analysis following your system instructions.
Reference evidence by its 8-character ID prefix when making claims.
""".strip()

        try:
            output, llm_resp = await self._call(user_message, FundamentalAgentOutput)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            agent_execution_histogram.labels(agent=self.role.value, status="success").observe(
                elapsed_ms / 1000
            )

            evidence_ids = [ev.id for ev in evidence]

            return AgentReport(
                agent=self.role,
                research_id=research_id,
                findings=output.findings,
                risks=output.risks,
                confidence=output.confidence,
                summary=output.summary,
                evidence_ids=evidence_ids,
                execution_time_ms=elapsed_ms,
                token_usage={
                    "input": llm_resp.input_tokens,
                    "output": llm_resp.output_tokens,
                },
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.error("fundamental_agent_error", research_id=research_id, error=str(exc))
            return self._failed_report(research_id, str(exc), elapsed_ms)
