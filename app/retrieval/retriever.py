"""
RAG Retrieval Pipeline

Architecture:

    query string
        → query embedding
        → DenseRetriever   (pgvector cosine similarity)
        → LexicalRetriever (PostgreSQL full-text search, BM25-like tsvector)
        → HybridMerger     (RRF: Reciprocal Rank Fusion)
        → MetadataFilter   (ticker, document_type, date range)
        → MMRReranker      (Maximum Marginal Relevance)
        → top-k Evidence objects

Why each layer exists:
- Dense: captures semantic similarity, handles synonyms and paraphrasing
- Lexical: captures exact keyword matches (ticker symbols, financial terms, numbers)
- RRF fusion: combines both without requiring score normalization
- MMR: reduces redundancy so the LLM doesn't see 6 near-identical paragraphs

See docs/RAG.md for detailed explanation.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import DocumentChunk, DocumentType, Evidence
from app.llm.provider import get_llm_provider
from app.observability.metrics import retrieval_chunk_counter, retrieval_latency_histogram

log = structlog.get_logger(__name__)


@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    method: str   # "dense" | "lexical" | "hybrid"


# ── Dense Retrieval ───────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Cosine-similarity search against pgvector embeddings.

    The actual SQL query is in the repository layer; this class
    assembles the query embedding and delegates to it.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def retrieve(
        self,
        query: str,
        ticker: str,
        top_k: int,
        document_types: list[DocumentType] | None = None,
    ) -> list[ScoredChunk]:
        t0 = time.monotonic()
        provider = get_llm_provider()
        [query_embedding] = await provider.embed([query])

        # Import here to avoid circular dependency
        from app.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(self._session)
        results = await repo.search_by_embedding(
            embedding=query_embedding,
            ticker=ticker,
            top_k=top_k,
            document_types=document_types,
        )

        latency = time.monotonic() - t0
        retrieval_latency_histogram.labels(method="dense").observe(latency)
        retrieval_chunk_counter.labels(method="dense").inc(len(results))

        return [ScoredChunk(chunk=chunk, score=score, method="dense")
                for chunk, score in results]


# ── Lexical Retrieval ─────────────────────────────────────────────────────────

class LexicalRetriever:
    """
    PostgreSQL full-text search using tsvector/tsquery.

    Approximates BM25 ranking by using ts_rank_cd with normalization.
    Falls back gracefully if no full-text index matches.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def retrieve(
        self,
        query: str,
        ticker: str,
        top_k: int,
        document_types: list[DocumentType] | None = None,
    ) -> list[ScoredChunk]:
        t0 = time.monotonic()
        from app.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(self._session)
        results = await repo.search_by_fulltext(
            query=query,
            ticker=ticker,
            top_k=top_k,
            document_types=document_types,
        )

        latency = time.monotonic() - t0
        retrieval_latency_histogram.labels(method="lexical").observe(latency)
        retrieval_chunk_counter.labels(method="lexical").inc(len(results))

        return [ScoredChunk(chunk=chunk, score=score, method="lexical")
                for chunk, score in results]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *ranked_lists: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """
    RRF score = Σ 1/(k + rank) across all lists.

    k=60 is the standard constant from Cormack et al. (2009).
    We use chunk ID as the merge key.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, ScoredChunk] = {}

    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            cid = scored.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks_by_id[cid] = scored

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        ScoredChunk(chunk=chunks_by_id[cid].chunk, score=score, method="hybrid")
        for cid, score in merged
    ]


# ── MMR Reranking ─────────────────────────────────────────────────────────────

def mmr_rerank(
    candidates: list[ScoredChunk],
    query_embedding: list[float],
    final_k: int,
    lambda_: float = 0.6,
) -> list[ScoredChunk]:
    """
    Maximum Marginal Relevance reranking.

    Iteratively selects the chunk that maximises:
        λ · relevance(chunk, query) − (1−λ) · max_similarity(chunk, selected)

    λ=1 → pure relevance (dense retrieval)
    λ=0 → pure diversity
    λ=0.6 balances both.

    This prevents the LLM from receiving 6 near-identical paragraphs
    when multiple chunks repeat the same fact.
    """
    if not candidates:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    def cos_sim(a: list[float], b: list[float]) -> float:
        na = np.array(a, dtype=np.float32)
        nb = np.array(b, dtype=np.float32)
        return float(np.dot(na, nb) / (np.linalg.norm(na) * np.linalg.norm(nb) + 1e-10))

    # Relevance to query (normalised)
    relevances = [
        cos_sim(c.chunk.embedding or [], list(q_norm))
        for c in candidates
    ]

    selected: list[ScoredChunk] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(final_k, len(candidates))):
        if not remaining:
            break

        best_idx = -1
        best_score = float("-inf")

        for i in remaining:
            rel = relevances[i]
            if not selected:
                max_sim = 0.0
            else:
                max_sim = max(
                    cos_sim(candidates[i].chunk.embedding or [], s.chunk.embedding or [])
                    for s in selected
                )
            score = lambda_ * rel - (1 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append(ScoredChunk(
                chunk=candidates[best_idx].chunk,
                score=best_score,
                method="mmr",
            ))
            remaining.remove(best_idx)

    return selected


# ── Evidence Packaging ────────────────────────────────────────────────────────

async def pack_evidence(
    scored_chunks: list[ScoredChunk],
    session: AsyncSession,
) -> list[Evidence]:
    """Convert scored chunks into Evidence objects with document metadata."""
    from app.repositories.document_repository import DocumentRepository
    doc_repo = DocumentRepository(session)

    evidence: list[Evidence] = []
    for sc in scored_chunks:
        chunk = sc.chunk
        doc = await doc_repo.get_by_id(chunk.document_id)
        if doc is None:
            continue
        evidence.append(Evidence(
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            source_filename=doc.filename,
            document_type=doc.metadata.document_type,
            published_date=doc.metadata.published_date,
            quote=chunk.text[:500],   # excerpt for citation
            relevance_score=max(0.0, min(1.0, sc.score)),
            retrieval_method=sc.method,
        ))

    return evidence


# ── Hybrid Retriever (Public Interface) ───────────────────────────────────────

class HybridRetriever:
    """
    Public entry point for the RAG retrieval pipeline.

    Combines DenseRetriever + LexicalRetriever via RRF,
    then applies MMR reranking for diversity.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._dense = DenseRetriever(session)
        self._lexical = LexicalRetriever(session)

    async def retrieve(
        self,
        query: str,
        ticker: str,
        *,
        document_types: list[DocumentType] | None = None,
        final_k: int | None = None,
        top_k: int | None = None,
        lambda_: float | None = None,
    ) -> list[Evidence]:
        from app.config.settings import get_settings
        settings = get_settings()
        _top_k = top_k or settings.retrieval_top_k
        _final_k = final_k or settings.retrieval_final_k
        _lambda = lambda_ if lambda_ is not None else settings.mmr_lambda

        t0 = time.monotonic()

        # Concurrent dense + lexical retrieval
        dense_task = self._dense.retrieve(query, ticker, _top_k, document_types)
        lexical_task = self._lexical.retrieve(query, ticker, _top_k, document_types)
        dense_results, lexical_results = await asyncio.gather(dense_task, lexical_task)

        # Fuse with RRF
        fused = reciprocal_rank_fusion(dense_results, lexical_results)

        # Need query embedding for MMR
        provider = get_llm_provider()
        [q_emb] = await provider.embed([query])

        # MMR reranking
        reranked = mmr_rerank(fused, q_emb, _final_k, _lambda)

        latency = time.monotonic() - t0
        retrieval_latency_histogram.labels(method="mmr").observe(latency)

        log.info(
            "retrieval_complete",
            query_preview=query[:60],
            ticker=ticker,
            dense_count=len(dense_results),
            lexical_count=len(lexical_results),
            fused_count=len(fused),
            final_count=len(reranked),
            latency_ms=int(latency * 1000),
        )

        return await pack_evidence(reranked, self._session)
