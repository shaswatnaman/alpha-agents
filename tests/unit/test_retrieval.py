"""Tests for the retrieval pipeline components."""

from __future__ import annotations

from app.domain.models import DocumentChunk
from app.retrieval.retriever import ScoredChunk, mmr_rerank, reciprocal_rank_fusion


def _make_chunk(id: str, embedding: list[float]) -> ScoredChunk:
    chunk = DocumentChunk(
        id=id,
        document_id="doc-1",
        chunk_index=0,
        text=f"text-{id}",
        char_start=0,
        char_end=100,
        embedding=embedding,
    )
    return ScoredChunk(chunk=chunk, score=1.0, method="dense")


class TestReciprocalRankFusion:
    def test_returns_union_of_lists(self) -> None:
        list_a = [_make_chunk("a", [1.0, 0.0]), _make_chunk("b", [0.9, 0.1])]
        list_b = [_make_chunk("c", [0.8, 0.2]), _make_chunk("a", [1.0, 0.0])]
        result = reciprocal_rank_fusion(list_a, list_b)
        ids = [r.chunk.id for r in result]
        assert set(ids) == {"a", "b", "c"}

    def test_duplicate_gets_higher_score(self) -> None:
        # "a" appears in both lists; it should rank first
        list_a = [_make_chunk("a", [1.0, 0.0]), _make_chunk("b", [0.5, 0.5])]
        list_b = [_make_chunk("a", [1.0, 0.0]), _make_chunk("c", [0.3, 0.7])]
        result = reciprocal_rank_fusion(list_a, list_b)
        assert result[0].chunk.id == "a"

    def test_empty_lists(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

    def test_method_set_to_hybrid(self) -> None:
        list_a = [_make_chunk("x", [1.0, 0.0])]
        result = reciprocal_rank_fusion(list_a)
        assert result[0].method == "hybrid"


class TestMMRRerank:
    def test_returns_at_most_k_results(self) -> None:
        chunks = [_make_chunk(str(i), [float(i) / 10, 1.0 - float(i) / 10]) for i in range(10)]
        query_emb = [1.0, 0.0]
        result = mmr_rerank(chunks, query_emb, final_k=3)
        assert len(result) <= 3

    def test_empty_candidates(self) -> None:
        assert mmr_rerank([], [1.0, 0.0], final_k=5) == []

    def test_returns_all_when_k_exceeds_candidates(self) -> None:
        chunks = [_make_chunk("a", [1.0, 0.0]), _make_chunk("b", [0.0, 1.0])]
        result = mmr_rerank(chunks, [1.0, 0.0], final_k=10)
        assert len(result) == 2

    def test_diversity_reduces_similar_chunks(self) -> None:
        # All chunks are nearly identical → MMR should still not return duplicates
        emb = [1.0, 0.0]
        chunks = [_make_chunk(str(i), emb[:]) for i in range(5)]
        result = mmr_rerank(chunks, [1.0, 0.0], final_k=5, lambda_=0.5)
        ids = [r.chunk.id for r in result]
        assert len(ids) == len(set(ids))  # no duplicates
