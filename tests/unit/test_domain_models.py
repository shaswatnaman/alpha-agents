"""Tests for domain model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models import (
    AgentFinding,
    AgentReport,
    AgentRole,
    ClaimType,
    ConfidenceScore,
    ResearchRequest,
)


class TestResearchRequest:
    def test_ticker_normalised_to_uppercase(self) -> None:
        req = ResearchRequest(ticker="aapl")
        assert req.ticker == "AAPL"

    def test_ticker_strips_whitespace(self) -> None:
        req = ResearchRequest(ticker="  MSFT  ")
        assert req.ticker == "MSFT"

    def test_default_status_is_pending(self) -> None:
        req = ResearchRequest(ticker="GOOG")
        assert req.status == "pending"


class TestConfidenceScore:
    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConfidenceScore(overall=1.5)

    def test_valid_confidence(self) -> None:
        cs = ConfidenceScore(overall=0.75)
        assert cs.overall == 0.75


class TestAgentFinding:
    def test_confidence_must_be_zero_to_one(self) -> None:
        with pytest.raises(ValidationError):
            AgentFinding(claim="test", claim_type=ClaimType.FACT, confidence=-0.1)

    def test_valid_finding(self) -> None:
        f = AgentFinding(claim="Revenue grew 20%", claim_type=ClaimType.FACT, confidence=0.9)
        assert f.confidence == 0.9


class TestAgentReport:
    def test_failed_report_has_zero_confidence(self) -> None:
        report = AgentReport(
            agent=AgentRole.FUNDAMENTAL,
            research_id="test-id",
            findings=[],
            confidence=0.0,
            summary="failed",
            failed=True,
            failure_reason="timeout",
        )
        assert report.failed is True
        assert report.confidence == 0.0
