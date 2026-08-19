"""
Technical Analyst Agent

Input: pre-computed TechnicalIndicators (never ask the LLM to compute numbers)
Output: structured AgentReport with market signal findings
"""
from __future__ import annotations

import time
from typing import Annotated

from pydantic import BaseModel, Field
import structlog

from app.agents.base import BaseAgent
from app.domain.models import (
    AgentFinding,
    AgentReport,
    AgentRole,
    ClaimType,
    RiskFactor,
    Sentiment,
    TechnicalIndicators,
)
from app.observability.metrics import agent_execution_histogram

log = structlog.get_logger(__name__)


class TechnicalAgentOutput(BaseModel):
    findings: list[AgentFinding]
    risks: list[RiskFactor]
    sentiment: Sentiment
    summary: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


def _format_indicators(ind: TechnicalIndicators) -> str:
    def pct(v: float | None) -> str:
        return f"{v*100:.1f}%" if v is not None else "N/A"

    def f(v: float | None, dec: int = 2) -> str:
        return f"{v:.{dec}f}" if v is not None else "N/A"

    return f"""
TECHNICAL INDICATORS for {ind.ticker} (as of {ind.as_of_date.date()}):

Price & Trend:
  Current Price:    {f(ind.current_price)}
  SMA-20:           {f(ind.sma_20)}
  SMA-50:           {f(ind.sma_50)}
  52-Week High:     {f(ind.price_52w_high)}
  52-Week Low:      {f(ind.price_52w_low)}

Momentum:
  RSI-14:           {f(ind.rsi_14, 1)}  (>70 overbought, <30 oversold)
  MACD Line:        {f(ind.macd, 4)}
  MACD Signal:      {f(ind.macd_signal, 4)}
  MACD Histogram:   {f(ind.macd_histogram, 4)}

Risk / Returns:
  Annualised Volatility: {pct(ind.annualized_volatility)}
  3M Annualised Return:  {pct(ind.annualized_return_3m)}
  20D Avg Volume:        {f(ind.volume_avg_20d, 0)}

Computation errors: {ind.computation_errors or 'None'}
""".strip()


class TechnicalAgent(BaseAgent):
    role = AgentRole.TECHNICAL
    prompt_file = "technical_analyst.txt"

    async def run(
        self,
        research_id: str,
        ticker: str,
        indicators: TechnicalIndicators,
    ) -> AgentReport:
        t0 = time.monotonic()

        # Technical agent has no document evidence — all data is computed
        user_message = f"""
COMPANY: {ticker}

{_format_indicators(indicators)}

Interpret these indicators according to your instructions.
Note: technical analysis has no document evidence — all findings have claim_type INFERENCE or UNCERTAINTY.
Set evidence_ids to an empty list for all findings.
""".strip()

        try:
            output, llm_resp = await self._call(user_message, TechnicalAgentOutput)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            agent_execution_histogram.labels(agent=self.role.value, status="success").observe(elapsed_ms / 1000)

            return AgentReport(
                agent=self.role,
                research_id=research_id,
                findings=output.findings,
                risks=output.risks,
                sentiment=output.sentiment,
                confidence=output.confidence,
                summary=output.summary,
                execution_time_ms=elapsed_ms,
                token_usage={"input": llm_resp.input_tokens, "output": llm_resp.output_tokens},
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.error("technical_agent_error", research_id=research_id, error=str(exc))
            return self._failed_report(research_id, str(exc), elapsed_ms)
