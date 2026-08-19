"""
Repository for document chunk retrieval — both vector and full-text.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import DocumentChunk, DocumentType


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Bulk insert chunks with their embeddings."""
        if not chunks:
            return
        stmt = text("""
            INSERT INTO chunks
                (id, document_id, chunk_index, text, char_start, char_end,
                 section_title, page_number, embedding, text_tsv, chunk_metadata)
            VALUES
                (:id, :document_id, :chunk_index, :text, :char_start, :char_end,
                 :section_title, :page_number, :embedding, to_tsvector('english', :text),
                 :chunk_metadata::jsonb)
            ON CONFLICT (id) DO NOTHING
        """)
        for chunk in chunks:
            await self._session.execute(
                stmt,
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "embedding": chunk.embedding,
                    "chunk_metadata": chunk.metadata,
                },
            )

    async def search_by_embedding(
        self,
        embedding: list[float],
        ticker: str,
        top_k: int,
        document_types: list[DocumentType] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Cosine similarity search via pgvector.

        Returns list of (chunk, similarity_score) ordered by descending score.
        """
        type_filter = ""
        params: dict[str, Any] = {
            "embedding": str(embedding),
            "ticker": ticker,
            "top_k": top_k,
        }

        if document_types:
            type_filter = "AND d.document_type = ANY(:doc_types)"
            params["doc_types"] = [
                dt.value if hasattr(dt, "value") else dt for dt in document_types
            ]

        stmt = text(f"""
            SELECT
                c.id, c.document_id, c.chunk_index, c.text,
                c.char_start, c.char_end, c.section_title, c.page_number,
                c.chunk_metadata,
                1 - (c.embedding <=> :embedding::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.ticker = :ticker
              {type_filter}
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        result = await self._session.execute(stmt, params)
        rows = result.fetchall()
        return [(_row_to_chunk(row), float(row.score)) for row in rows]

    async def search_by_fulltext(
        self,
        query: str,
        ticker: str,
        top_k: int,
        document_types: list[DocumentType] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        PostgreSQL tsvector full-text search with ts_rank_cd scoring.
        """
        type_filter = ""
        params: dict[str, Any] = {
            "query": query,
            "ticker": ticker,
            "top_k": top_k,
        }
        if document_types:
            type_filter = "AND d.document_type = ANY(:doc_types)"
            params["doc_types"] = [
                dt.value if hasattr(dt, "value") else dt for dt in document_types
            ]

        stmt = text(f"""
            SELECT
                c.id, c.document_id, c.chunk_index, c.text,
                c.char_start, c.char_end, c.section_title, c.page_number,
                c.chunk_metadata,
                ts_rank_cd(c.text_tsv, websearch_to_tsquery('english', :query)) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.ticker = :ticker
              {type_filter}
              AND c.text_tsv @@ websearch_to_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :top_k
        """)

        result = await self._session.execute(stmt, params)
        rows = result.fetchall()
        return [(_row_to_chunk(row), float(row.score)) for row in rows]


def _row_to_chunk(row: Any) -> DocumentChunk:
    return DocumentChunk(
        id=row.id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        text=row.text,
        char_start=row.char_start,
        char_end=row.char_end,
        section_title=row.section_title,
        page_number=row.page_number,
        metadata=dict(row.chunk_metadata) if row.chunk_metadata else {},
    )
