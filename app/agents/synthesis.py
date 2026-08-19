"""
Synthesis Agent

Integrates fundamental + technical + sentiment reports plus critic findings
into a final structured ResearchReport.
"""

from __future__ import annotations

import time
from typing import Annotated

import structlog
from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.domain.models import (
    AgentReport,
    AgentRole,
    ConflictingSignal,
    CriticFinding,
    Evidence,
    RiskFactor,
    Sentiment,
)
from app.observability.metrics import agent_execution_histogram

log = structlog.get_logger(__name__)


class SynthesisOutput(BaseModel):
    executive_summary: str
    fundamental_view: str
    technical_view: str
    sentiment_view: str
    key_risks: list[RiskFactor]
    conflicting_signals: list[ConflictingSignal]
    overall_sentiment: Sentiment
    conclusion: str
    confidence_overall: Annotated[float, Field(ge=0.0, le=1.0)]
    confidence_fundamental: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    confidence_technical: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    confidence_sentiment: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


def _report_summary(r: AgentReport) -> str:
    if r.failed:
        return f"[FAILED — {r.failure_reason}]"
    return (
        f"Confidence: {r.confidence:.2f}\n"
        f"Summary: {r.summary}\n"
        f"Sentiment: {r.sentiment or 'N/A'}\n"
        f"Risk count: {len(r.risks)}"
    )


def _format_critic(findings: list[CriticFinding]) -> str:
    if not findings:
        return "No critic findings."
    lines = []
    for cf in findings:
        lines.append(
            f'[{cf.severity}] {cf.affected_agent}: "{cf.affected_claim}"\n'
            f"  Issue: {cf.issue}\n"
            f"  Recommendation: {cf.recommendation}"
        )
    return "\n\n".join(lines)


class SynthesisAgent(BaseAgent):
    role = AgentRole.SYNTHESIS
    prompt_file = "synthesis_agent.txt"

    async def run(
        self,
        research_id: str,
        ticker: str,
        fundamental: AgentReport,
        technical: AgentReport,
        sentiment: AgentReport,
        critic_findings: list[CriticFinding],
        all_evidence: list[Evidence],
    ) -> AgentReport:
        t0 = time.monotonic()

        user_message = f"""
COMPANY: {ticker}
RESEARCH ID: {research_id}

═══ FUNDAMENTAL ANALYST ═══
{_report_summary(fundamental)}

═══ TECHNICAL ANALYST ═══
{_report_summary(technical)}

═══ SENTIMENT ANALYST ═══
{_report_summary(sentiment)}

═══ CRITIC REVIEW ═══
{_format_critic(critic_findings)}

Total evidence pieces available: {len(all_evidence)}

Synthesise all of the above into a research report.
""".strip()

        try:
            output, llm_resp = await self._call(user_message, SynthesisOutput)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            agent_execution_histogram.labels(agent=self.role.value, status="success").observe(
                elapsed_ms / 1000
            )

            from app.domain.models import AgentFinding, ClaimType

            report = AgentReport(
                agent=self.role,
                research_id=research_id,
                findings=[
                    AgentFinding(
                        claim=output.conclusion,
                        claim_type=ClaimType.INFERENCE,
                        confidence=output.confidence_overall,
                    )
                ],
                risks=output.key_risks,
                sentiment=output.overall_sentiment,
                confidence=output.confidence_overall,
                summary=output.executive_summary,
                execution_time_ms=elapsed_ms,
                token_usage={"input": llm_resp.input_tokens, "output": llm_resp.output_tokens},
            )

            # Attach synthesis output so orchestrator can build ResearchReport
            report._synthesis_output = output  # type: ignore[attr-defined]
            return report

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.error("synthesis_agent_error", research_id=research_id, error=str(exc))
            return self._failed_report(research_id, str(exc), elapsed_ms)
