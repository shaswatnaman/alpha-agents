from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Document, DocumentMetadata, DocumentType
from app.repositories.models import DocumentORM


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, document_id: str) -> Document | None:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.id == document_id)
        )
        row = result.scalar_one_or_none()
        return _orm_to_domain(row) if row else None

    async def get_by_hash(self, content_hash: str) -> Document | None:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.content_hash == content_hash)
        )
        row = result.scalar_one_or_none()
        return _orm_to_domain(row) if row else None

    async def list_by_ticker(self, ticker: str) -> list[Document]:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.ticker == ticker).order_by(DocumentORM.created_at.desc())
        )
        return [_orm_to_domain(row) for row in result.scalars().all()]

    async def insert(self, doc: Document) -> None:
        orm = DocumentORM(
            id=doc.id,
            filename=doc.filename,
            content_hash=doc.content_hash,
            ticker=doc.metadata.ticker,
            document_type=doc.metadata.document_type,
            source_url=doc.metadata.source_url,
            published_date=doc.metadata.published_date,
            fiscal_year=doc.metadata.fiscal_year,
            fiscal_quarter=doc.metadata.fiscal_quarter,
        )
        self._session.add(orm)


def _orm_to_domain(row: DocumentORM) -> Document:
    return Document(
        id=row.id,
        filename=row.filename,
        content_hash=row.content_hash,
        metadata=DocumentMetadata(
            ticker=row.ticker,
            document_type=DocumentType(row.document_type),
            source_url=row.source_url,
            published_date=row.published_date,
            fiscal_year=row.fiscal_year,
            fiscal_quarter=row.fiscal_quarter,
        ),
        created_at=row.created_at,
    )
