"""
Sentiment Analyst Agent

Input: list of fetched NewsArticle objects (never hallucinates news)
Output: structured AgentReport with sentiment findings
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Annotated

import structlog
from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.data.news_fetcher import NewsArticle
from app.domain.models import (
    AgentFinding,
    AgentReport,
    AgentRole,
    ClaimType,
    Evidence,
    RiskFactor,
    Sentiment,
)
from app.observability.metrics import agent_execution_histogram

log = structlog.get_logger(__name__)

MAX_ARTICLE_BODY_CHARS = 800


class SentimentAgentOutput(BaseModel):
    findings: list[AgentFinding]
    risks: list[RiskFactor]
    sentiment: Sentiment
    summary: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    key_events: list[str] = Field(default_factory=list)


def _format_articles(articles: list[NewsArticle]) -> str:
    if not articles:
        return "No news articles available."
    lines = []
    for i, art in enumerate(articles, 1):
        date_str = art.published_at.strftime("%Y-%m-%d") if art.published_at else "unknown date"
        body_preview = art.body[:MAX_ARTICLE_BODY_CHARS].replace("\n", " ") if art.body else "(no body)"
        lines.append(
            f"Article {i} [{date_str}] — {art.source}\n"
            f"Title: {art.title}\n"
            f"URL: {art.url}\n"
            f"Content: {body_preview}"
        )
    return "\n\n---\n\n".join(lines)


class SentimentAgent(BaseAgent):
    role = AgentRole.SENTIMENT
    prompt_file = "sentiment_analyst.txt"

    async def run(
        self,
        research_id: str,
        ticker: str,
        articles: list[NewsArticle],
    ) -> AgentReport:
        t0 = time.monotonic()

        if not articles:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return AgentReport(
                agent=self.role,
                research_id=research_id,
                findings=[],
                confidence=0.0,
                summary="No news articles were available for sentiment analysis.",
                failed=False,   # not a failure — just no data
                execution_time_ms=elapsed_ms,
            )

        user_message = f"""
COMPANY: {ticker}
TOTAL ARTICLES: {len(articles)}

{_format_articles(articles)}

Analyse the sentiment according to your instructions.
When referencing a specific article, mention its number (Article N) in the claim.
Set evidence_ids to an empty list (news articles are not stored as document chunks).
""".strip()

        try:
            output, llm_resp = await self._call(user_message, SentimentAgentOutput)
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
            log.error("sentiment_agent_error", research_id=research_id, error=str(exc))
            return self._failed_report(research_id, str(exc), elapsed_ms)
