"""
Risk / Critic Agent

Reviews the three specialist agent outputs and challenges:
- Unsupported claims (no evidence IDs)
- Inter-agent contradictions
- Overconfident ratings on thin evidence
- Stale or missing data risks
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
    CriticFinding,
    Severity,
)
from app.observability.metrics import agent_execution_histogram

log = structlog.get_logger(__name__)


class CriticAgentOutput(BaseModel):
    critic_findings: list[CriticFinding]
    overall_risk_level: Severity
    summary: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


def _format_report(report: AgentReport) -> str:
    if report.failed:
        return f"[FAILED: {report.failure_reason}]"
    lines = [
        f"Agent: {report.agent}",
        f"Confidence: {report.confidence:.2f}",
        f"Summary: {report.summary}",
        "\nFindings:",
    ]
    for f in report.findings:
        lines.append(f"  [{f.claim_type}] {f.claim}  (evidence_ids: {f.evidence_ids or 'NONE'})")
    if report.risks:
        lines.append("\nRisks:")
        for r in report.risks:
            lines.append(f"  [{r.severity}] {r.description}")
    return "\n".join(lines)


class CriticAgent(BaseAgent):
    role = AgentRole.CRITIC
    prompt_file = "critic_agent.txt"

    async def run(
        self,
        research_id: str,
        fundamental: AgentReport,
        technical: AgentReport,
        sentiment: AgentReport,
    ) -> tuple[list[CriticFinding], AgentReport]:
        t0 = time.monotonic()

        user_message = f"""
Below are the outputs of three specialist agents. Review them critically.

═══ FUNDAMENTAL ANALYST ═══
{_format_report(fundamental)}

═══ TECHNICAL ANALYST ═══
{_format_report(technical)}

═══ SENTIMENT ANALYST ═══
{_format_report(sentiment)}

Apply your critic review instructions and produce a structured list of findings.
""".strip()

        try:
            output, llm_resp = await self._call(user_message, CriticAgentOutput)
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
                        claim=cf.issue,
                        claim_type=ClaimType.INFERENCE,
                        confidence=0.8,
                    )
                    for cf in output.critic_findings
                ],
                confidence=output.confidence,
                summary=output.summary,
                execution_time_ms=elapsed_ms,
                token_usage={"input": llm_resp.input_tokens, "output": llm_resp.output_tokens},
            )
            return output.critic_findings, report

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.error("critic_agent_error", research_id=research_id, error=str(exc))
            failed = self._failed_report(research_id, str(exc), elapsed_ms)
            return [], failed
