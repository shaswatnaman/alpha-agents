"""
Research Orchestrator — coordinates the full multi-agent pipeline.

Execution order:
    1. Parallel: FundamentalAgent, TechnicalAgent, SentimentAgent
       (These have no dependency on each other → safe to run concurrently)
    2. Sequential: CriticAgent
       (Requires all three reports above)
    3. Sequential: SynthesisAgent
       (Requires reports + critic findings)

Concurrency safety:
    - asyncio.gather() for step 1: safe because each agent operates on
      independent inputs (different context, different LLM calls)
    - Redis/DB writes per agent happen after the gather returns
    - No shared mutable state during parallel execution

Failure handling:
    - A failed agent produces a failed AgentReport (never raises)
    - The pipeline continues with remaining agents
    - Final report is marked PARTIAL if any agent failed
"""

from __future__ import annotations

import asyncio
import time

import structlog

from app.agents.critic import CriticAgent
from app.agents.fundamental import FundamentalAgent
from app.agents.sentiment import SentimentAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.technical import TechnicalAgent
from app.cache.idempotency import get_idempotent_result, store_idempotent_result
from app.cache.redis_client import get_redis
from app.config.settings import get_settings
from app.data.market_data import fetch_fundamental_metrics, fetch_technical_indicators
from app.data.news_fetcher import fetch_news
from app.domain.models import (
    AgentRole,
    Citation,
    ConfidenceScore,
    Evidence,
    ResearchReport,
    ResearchRequest,
    ResearchStatus,
    Sentiment,
)
from app.observability.metrics import (
    active_research_gauge,
    research_counter,
    research_latency_histogram,
)
from app.repositories.research_repository import ResearchRepository
from app.retrieval.retriever import HybridRetriever

log = structlog.get_logger(__name__)


class ResearchOrchestrator:
    def __init__(self, research_repo: ResearchRepository) -> None:
        self._repo = research_repo

    async def start_research(self, request: ResearchRequest) -> str:
        """
        Enqueue a research job and return the research_id immediately.
        The actual pipeline runs as a background task.
        """
        # Idempotency check — if this key was already processed, return existing ID
        if request.idempotency_key:
            existing_id = await get_idempotent_result(request.idempotency_key)
            if existing_id:
                log.info("idempotency_return", research_id=existing_id)
                return existing_id

        # Persist the request immediately (status=PENDING)
        await self._repo.create_request(request)
        log.info("research_created", research_id=request.id, ticker=request.ticker)

        # Store idempotency mapping
        if request.idempotency_key:
            await store_idempotent_result(request.idempotency_key, request.id)

        # Schedule background execution (does not block response)
        asyncio.create_task(self._run_pipeline(request))

        return request.id

    async def _run_pipeline(self, request: ResearchRequest) -> None:
        """Full pipeline — runs as a background asyncio task."""
        t0 = time.monotonic()
        active_research_gauge.inc()
        redis = await get_redis()
        research_id = request.id
        ticker = request.ticker

        async def _set_status(status_key: str) -> None:
            await redis.setex(
                f"research:status:{research_id}",
                3600,
                status_key,
            )

        try:
            await self._repo.update_request_status(research_id, ResearchStatus.RUNNING)
            await _set_status("running:data_fetch")

            # ── Step 1: Gather inputs concurrently ──────────────────────────
            log.info("pipeline_step", step="data_fetch", research_id=research_id)
            technical_indicators, fundamental_metrics, news_articles = await asyncio.gather(
                fetch_technical_indicators(ticker),
                fetch_fundamental_metrics(ticker),
                fetch_news(ticker),
                return_exceptions=True,
            )
            # Replace exceptions with empty/error objects
            if isinstance(technical_indicators, Exception):
                from datetime import datetime

                from app.domain.models import TechnicalIndicators

                technical_indicators = TechnicalIndicators(
                    ticker=ticker,
                    as_of_date=datetime.utcnow(),
                    computation_errors=[str(technical_indicators)],
                )
            if isinstance(fundamental_metrics, Exception):
                from datetime import datetime

                from app.domain.models import FundamentalMetrics

                fundamental_metrics = FundamentalMetrics(
                    ticker=ticker,
                    as_of_date=datetime.utcnow(),
                    fetch_errors=[str(fundamental_metrics)],
                )
            if isinstance(news_articles, Exception):
                news_articles = []

            # ── Step 2: RAG retrieval for fundamental analysis ───────────────
            await _set_status("running:retrieval")
            log.info("pipeline_step", step="retrieval", research_id=research_id)

            # Retrieval happens inside the same session scope as the pipeline
            # We pass the session from the outer scope via the repository's session
            from app.repositories.database import get_db_context

            async with get_db_context() as session:
                retriever = HybridRetriever(session)

                fundamental_queries = [
                    f"{ticker} revenue growth earnings",
                    f"{ticker} operating margin profitability",
                    f"{ticker} balance sheet debt cash",
                    f"{ticker} business strategy risks",
                ]
                evidence_sets = await asyncio.gather(
                    *[retriever.retrieve(q, ticker) for q in fundamental_queries]
                )

            # Deduplicate evidence by chunk_id
            seen_chunk_ids: set[str] = set()
            all_evidence: list[Evidence] = []
            for ev_list in evidence_sets:
                for ev in ev_list:
                    if ev.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(ev.chunk_id)
                        all_evidence.append(ev)

            log.info(
                "retrieval_complete", research_id=research_id, evidence_count=len(all_evidence)
            )

            # ── Step 3: Run specialist agents in parallel ────────────────────
            await _set_status("running:agents")
            log.info("pipeline_step", step="agents", research_id=research_id)

            fundamental_agent = FundamentalAgent()
            technical_agent = TechnicalAgent()
            sentiment_agent = SentimentAgent()

            fundamental_report, technical_report, sentiment_report = await asyncio.gather(
                fundamental_agent.run(research_id, ticker, fundamental_metrics, all_evidence),
                technical_agent.run(research_id, ticker, technical_indicators),
                sentiment_agent.run(research_id, ticker, news_articles),
            )

            # Persist agent runs (sequential — after gather)
            async with get_db_context() as session:
                repo = ResearchRepository(session)
                for rpt in (fundamental_report, technical_report, sentiment_report):
                    await repo.save_agent_run(rpt)

            # ── Step 4: Critic agent (sequential dependency) ─────────────────
            await _set_status("running:critic")
            log.info("pipeline_step", step="critic", research_id=research_id)

            critic_agent = CriticAgent()
            critic_findings, critic_report = await critic_agent.run(
                research_id, fundamental_report, technical_report, sentiment_report
            )

            async with get_db_context() as session:
                repo = ResearchRepository(session)
                await repo.save_agent_run(critic_report)

            # ── Step 5: Synthesis agent ──────────────────────────────────────
            await _set_status("running:synthesis")
            log.info("pipeline_step", step="synthesis", research_id=research_id)

            synthesis_agent = SynthesisAgent()
            synthesis_report = await synthesis_agent.run(
                research_id,
                ticker,
                fundamental_report,
                technical_report,
                sentiment_report,
                critic_findings,
                all_evidence,
            )

            async with get_db_context() as session:
                repo = ResearchRepository(session)
                await repo.save_agent_run(synthesis_report)

            # ── Step 6: Assemble final ResearchReport ────────────────────────
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            synth_out = getattr(synthesis_report, "_synthesis_output", None)

            total_tokens = sum(
                sum(r.token_usage.values())
                for r in [
                    fundamental_report,
                    technical_report,
                    sentiment_report,
                    critic_report,
                    synthesis_report,
                ]
            )
            any_failed = any(
                r.failed
                for r in [
                    fundamental_report,
                    technical_report,
                    sentiment_report,
                    critic_report,
                    synthesis_report,
                ]
            )

            # Build citations: link each evidence item to the claims that reference it
            evidence_by_id = {ev.id: ev for ev in all_evidence}
            citations: list[Citation] = []
            for finding in (
                fundamental_report.findings + technical_report.findings + sentiment_report.findings
            ):
                cited_ev = [
                    evidence_by_id[eid] for eid in finding.evidence_ids if eid in evidence_by_id
                ]
                if cited_ev:
                    citations.append(Citation(claim=finding.claim, evidence=cited_ev))

            report = ResearchReport(
                research_id=research_id,
                ticker=ticker,
                executive_summary=synth_out.executive_summary
                if synth_out
                else synthesis_report.summary,
                fundamental_view=synth_out.fundamental_view if synth_out else "Unavailable",
                technical_view=synth_out.technical_view if synth_out else "Unavailable",
                sentiment_view=synth_out.sentiment_view if synth_out else "Unavailable",
                key_risks=synth_out.key_risks if synth_out else [],
                conflicting_signals=synth_out.conflicting_signals if synth_out else [],
                citations=citations,
                confidence=ConfidenceScore(
                    overall=synth_out.confidence_overall
                    if synth_out
                    else synthesis_report.confidence,
                    fundamental=synth_out.confidence_fundamental
                    if synth_out
                    else fundamental_report.confidence,
                    technical=synth_out.confidence_technical
                    if synth_out
                    else technical_report.confidence,
                    sentiment=synth_out.confidence_sentiment
                    if synth_out
                    else sentiment_report.confidence,
                    data_completeness=1.0
                    - (
                        0.2
                        * len(
                            [
                                r
                                for r in [fundamental_report, technical_report, sentiment_report]
                                if r.failed
                            ]
                        )
                    ),
                ),
                overall_sentiment=synth_out.overall_sentiment if synth_out else Sentiment.NEUTRAL,
                critic_findings=critic_findings,
                agent_reports={
                    AgentRole.FUNDAMENTAL.value: fundamental_report,
                    AgentRole.TECHNICAL.value: technical_report,
                    AgentRole.SENTIMENT.value: sentiment_report,
                    AgentRole.CRITIC.value: critic_report,
                    AgentRole.SYNTHESIS.value: synthesis_report,
                },
                total_execution_time_ms=elapsed_ms,
                total_tokens=total_tokens,
            )

            # Cache the report in Redis for fast subsequent reads
            redis_key = f"research:report:{research_id}"
            await redis.setex(
                redis_key,
                get_settings().research_cache_ttl_seconds,
                report.model_dump_json(),
            )

            # Persist to PostgreSQL
            async with get_db_context() as session:
                repo = ResearchRepository(session)
                await repo.save_report(report)
                status = ResearchStatus.PARTIAL if any_failed else ResearchStatus.COMPLETED
                await repo.update_request_status(research_id, status, report_id=report.id)

            elapsed_total = time.monotonic() - t0
            research_counter.labels(status=status.value).inc()
            research_latency_histogram.observe(elapsed_total)
            await _set_status(f"completed:{report.id}")

            log.info(
                "pipeline_complete",
                research_id=research_id,
                ticker=ticker,
                elapsed_ms=elapsed_ms,
                total_tokens=total_tokens,
                status=status.value,
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.error("pipeline_error", research_id=research_id, error=str(exc), exc_info=True)
            research_counter.labels(status="failed").inc()
            await _set_status(f"failed:{exc}")
            async with get_db_context() as session:
                repo = ResearchRepository(session)
                await repo.update_request_status(
                    research_id, ResearchStatus.FAILED, error_message=str(exc)
                )
        finally:
            active_research_gauge.dec()
