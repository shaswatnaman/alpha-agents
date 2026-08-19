"""
Unit tests for agent classes — LLM is fully mocked.
Tests verify prompt construction, structured output parsing, and failure handling.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from app.agents.fundamental import FundamentalAgent
from app.agents.sentiment import SentimentAgent
from app.agents.technical import TechnicalAgent
from app.domain.models import (
    AgentRole,
    FundamentalMetrics,
    TechnicalIndicators,
)
from app.llm.provider import LLMResponse


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=100,
        output_tokens=50,
        model="gpt-4o-mini",
        latency_ms=200,
        estimated_cost_usd=0.001,
    )


@pytest.fixture
def fundamental_metrics() -> FundamentalMetrics:
    return FundamentalMetrics(
        ticker="AAPL",
        as_of_date=datetime(2024, 1, 1),
        market_cap=3e12,
        pe_ratio=28.0,
        revenue_ttm=400e9,
        gross_margin=0.44,
        operating_margin=0.30,
    )


@pytest.fixture
def technical_indicators() -> TechnicalIndicators:
    return TechnicalIndicators(
        ticker="AAPL",
        as_of_date=datetime(2024, 1, 1),
        current_price=185.0,
        sma_20=180.0,
        sma_50=175.0,
        rsi_14=62.0,
        macd=1.5,
        macd_signal=1.0,
        macd_histogram=0.5,
        annualized_volatility=0.22,
        annualized_return_3m=0.18,
    )


class TestFundamentalAgent:
    @pytest.mark.asyncio
    async def test_run_returns_agent_report(self, fundamental_metrics: FundamentalMetrics) -> None:
        mock_output = {
            "findings": [
                {
                    "claim": "Revenue is strong.",
                    "claim_type": "fact",
                    "evidence_ids": [],
                    "confidence": 0.85,
                }
            ],
            "risks": [],
            "summary": "Strong fundamental picture.",
            "confidence": 0.85,
        }

        import json

        mock_provider = AsyncMock()
        mock_provider.embed = AsyncMock(return_value=[[0.1] * 10])
        mock_provider.complete = AsyncMock(return_value=_mock_llm_response(json.dumps(mock_output)))

        agent = FundamentalAgent(provider=mock_provider)
        report = await agent.run(
            research_id="test-id",
            ticker="AAPL",
            metrics=fundamental_metrics,
            evidence=[],
        )

        assert report.agent == AgentRole.FUNDAMENTAL
        assert not report.failed
        assert report.confidence == 0.85
        assert len(report.findings) == 1

    @pytest.mark.asyncio
    async def test_run_handles_llm_failure(self, fundamental_metrics: FundamentalMetrics) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        agent = FundamentalAgent(provider=mock_provider)
        report = await agent.run(
            research_id="test-id",
            ticker="AAPL",
            metrics=fundamental_metrics,
            evidence=[],
        )

        assert report.failed is True
        assert "LLM timeout" in (report.failure_reason or "")
        assert report.confidence == 0.0


class TestTechnicalAgent:
    @pytest.mark.asyncio
    async def test_run_returns_agent_report(
        self, technical_indicators: TechnicalIndicators
    ) -> None:
        mock_output = {
            "findings": [
                {
                    "claim": "RSI suggests momentum.",
                    "claim_type": "inference",
                    "evidence_ids": [],
                    "confidence": 0.7,
                }
            ],
            "risks": [],
            "sentiment": "bullish",
            "summary": "Mildly bullish trend.",
            "confidence": 0.7,
        }

        import json

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_mock_llm_response(json.dumps(mock_output)))

        agent = TechnicalAgent(provider=mock_provider)
        report = await agent.run(
            research_id="test-id",
            ticker="AAPL",
            indicators=technical_indicators,
        )

        assert not report.failed
        assert report.sentiment == "bullish"

    @pytest.mark.asyncio
    async def test_run_handles_failure(self, technical_indicators: TechnicalIndicators) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=Exception("API error"))

        agent = TechnicalAgent(provider=mock_provider)
        report = await agent.run(
            research_id="test-id",
            ticker="AAPL",
            indicators=technical_indicators,
        )

        assert report.failed is True


class TestSentimentAgent:
    @pytest.mark.asyncio
    async def test_returns_empty_report_when_no_articles(self) -> None:
        mock_provider = AsyncMock()
        agent = SentimentAgent(provider=mock_provider)
        report = await agent.run(
            research_id="test-id",
            ticker="AAPL",
            articles=[],
        )
        assert not report.failed
        assert report.confidence == 0.0
        assert "No news" in report.summary
        mock_provider.complete.assert_not_called()
