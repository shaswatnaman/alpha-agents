"""
Document ingestion endpoint — upload PDFs/TXTs for RAG.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.rate_limiter import api_limiter
from app.config.settings import get_settings
from app.domain.models import DocumentMetadata, DocumentType
from app.ingestion.pipeline import ingest_document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.database import get_db
from app.repositories.document_repository import DocumentRepository
from app.schemas.api import DocumentUploadResponse

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


def _get_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    settings = get_settings()
    if x_api_key not in settings.api_keys_set:
        raise HTTPException(status_code=401, detail={"error": "invalid_api_key"})
    return x_api_key


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    document_type: str = Form(default="other"),
    fiscal_year: int | None = Form(default=None),
    fiscal_quarter: int | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(_get_api_key),
) -> DocumentUploadResponse:
    await api_limiter.check(api_key)

    settings = get_settings()
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in settings.allowed_document_types:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_file_type", "message": f"Supported types: {settings.allowed_document_types}"},
        )

    data = await file.read()
    if len(data) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "message": f"Max {settings.max_upload_size_mb}MB"},
        )

    # Check for duplicate (content hash)
    doc_repo = DocumentRepository(db)
    import hashlib
    content_hash = hashlib.sha256(data).hexdigest()
    existing = await doc_repo.get_by_hash(content_hash)
    if existing:
        log.info("document_already_exists", document_id=existing.id)
        return DocumentUploadResponse(
            document_id=existing.id,
            filename=filename,
            chunk_count=0,
            already_existed=True,
            message="Document already ingested.",
        )

    try:
        doc_type = DocumentType(document_type)
    except ValueError:
        doc_type = DocumentType.OTHER

    metadata = DocumentMetadata(
        ticker=ticker.strip().upper(),
        document_type=doc_type,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
    )

    doc, chunks = await ingest_document(data, filename, metadata)

    # Persist document + chunks
    await doc_repo.insert(doc)
    chunk_repo = ChunkRepository(db)
    await chunk_repo.insert_chunks(chunks)

    log.info("document_ingested", document_id=doc.id, chunk_count=len(chunks), ticker=ticker)

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=filename,
        chunk_count=len(chunks),
        already_existed=False,
        message=f"Successfully ingested {len(chunks)} chunks.",
    )


@router.get("/{ticker}", summary="List documents for a ticker")
async def list_documents(
    ticker: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(_get_api_key),
) -> list[dict]:
    await api_limiter.check(api_key)
    repo = DocumentRepository(db)
    docs = await repo.list_by_ticker(ticker.upper())
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "document_type": d.metadata.document_type,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]
