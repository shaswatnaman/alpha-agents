"""Tests for evaluation metric functions."""
from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    agent_agreement_rate,
    citation_accuracy,
    hallucination_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial(self) -> None:
        assert precision_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(2 / 3)

    def test_k_zero(self) -> None:
        assert precision_at_k(["a", "b"], ["a"], k=0) == 0.0


class TestRecallAtK:
    def test_all_retrieved(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_none_retrieved(self) -> None:
        assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial(self) -> None:
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == pytest.approx(0.5)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], [], k=2) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_no_relevant(self) -> None:
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_lower_score_for_later_rank(self) -> None:
        # "a" at rank 1 should score higher than "a" at rank 2
        score_rank1 = ndcg_at_k(["a", "x"], ["a"], k=2)
        score_rank2 = ndcg_at_k(["x", "a"], ["a"], k=2)
        assert score_rank1 > score_rank2


class TestCitationAccuracy:
    def test_all_correct(self) -> None:
        citations = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        assert citation_accuracy(citations, {"c1", "c2", "c3"}) == 1.0

    def test_none_correct(self) -> None:
        citations = [{"chunk_id": "x"}]
        assert citation_accuracy(citations, {"c1", "c2"}) == 0.0

    def test_empty_citations(self) -> None:
        assert citation_accuracy([], {"c1"}) == 0.0


class TestHallucinationRate:
    def test_no_facts(self) -> None:
        findings = [{"claim_type": "inference", "evidence_ids": []}]
        assert hallucination_rate(findings, set()) == 0.0

    def test_all_facts_have_evidence(self) -> None:
        findings = [{"claim_type": "fact", "evidence_ids": ["e1"]}]
        assert hallucination_rate(findings, {"e1"}) == 0.0

    def test_unsupported_fact(self) -> None:
        findings = [{"claim_type": "fact", "evidence_ids": []}]
        assert hallucination_rate(findings, set()) == 1.0


class TestAgentAgreementRate:
    def test_all_agree(self) -> None:
        reports = [
            {"failed": False, "sentiment": "bullish"},
            {"failed": False, "sentiment": "bullish"},
        ]
        assert agent_agreement_rate(reports) == 1.0

    def test_all_disagree(self) -> None:
        reports = [
            {"failed": False, "sentiment": "bullish"},
            {"failed": False, "sentiment": "bearish"},
        ]
        assert agent_agreement_rate(reports) == 0.0

    def test_ignores_failed_agents(self) -> None:
        reports = [
            {"failed": True, "sentiment": "bearish"},
            {"failed": False, "sentiment": "bullish"},
            {"failed": False, "sentiment": "bullish"},
        ]
        assert agent_agreement_rate(reports) == 1.0
