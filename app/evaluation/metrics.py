"""
Evaluation metric computation.

All metrics are computed deterministically from ground-truth labels.
We NEVER claim improvements without measured data.
"""
from __future__ import annotations

import math
from typing import Any


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of all relevant items found in top-k."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain.

    Relevance is binary: 1 if in relevant set, 0 otherwise.
    Measures both relevance and rank order quality.
    """
    if not relevant or k == 0:
        return 0.0
    relevant_set = set(relevant)

    def dcg(items: list[str]) -> float:
        return sum(
            (1.0 / math.log2(i + 2))
            for i, item in enumerate(items[:k])
            if item in relevant_set
        )

    idcg = dcg(list(relevant_set)[:k])
    return dcg(retrieved) / idcg if idcg > 0 else 0.0


def citation_accuracy(
    citations: list[dict[str, Any]],
    ground_truth_chunk_ids: set[str],
) -> float:
    """
    Fraction of cited chunk_ids that appear in the ground truth set.

    A citation is 'accurate' if the referenced chunk actually supports
    the claim (approximated by presence in the ground truth set).
    """
    if not citations:
        return 0.0
    cited_ids = {c.get("chunk_id", "") for c in citations}
    if not cited_ids:
        return 0.0
    correct = cited_ids & ground_truth_chunk_ids
    return len(correct) / len(cited_ids)


def hallucination_rate(
    findings: list[dict[str, Any]],
    known_facts: set[str],
) -> float:
    """
    Fraction of FACT-type findings that have no evidence_ids.

    A finding with claim_type='fact' but no evidence is a potential hallucination.
    This is a conservative proxy — actual hallucination detection requires
    claim-by-claim verification against source text.
    """
    fact_findings = [f for f in findings if f.get("claim_type") == "fact"]
    if not fact_findings:
        return 0.0
    unsupported = sum(1 for f in fact_findings if not f.get("evidence_ids"))
    return unsupported / len(fact_findings)


def agent_agreement_rate(reports: list[dict[str, Any]]) -> float:
    """
    Fraction of agent pairs whose sentiment values agree.

    Only considers non-failed agents with a non-null sentiment.
    """
    sentiments = [
        r.get("sentiment")
        for r in reports
        if not r.get("failed") and r.get("sentiment") is not None
    ]
    if len(sentiments) < 2:
        return 1.0  # single agent, trivially "agrees with itself"
    pairs = [(sentiments[i], sentiments[j]) for i in range(len(sentiments)) for j in range(i + 1, len(sentiments))]
    agreed = sum(1 for a, b in pairs if a == b)
    return agreed / len(pairs)
