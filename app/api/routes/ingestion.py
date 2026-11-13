from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_ingestion_service
from app.core.config import get_settings
from app.db.models import Document
from app.schemas.ingestion import DocumentResponse
from app.services.chunking import ChunkingStrategy
from app.services.ingestion import IngestionError, IngestionService

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/documents", response_model=DocumentResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    strategy: ChunkingStrategy = Form(ChunkingStrategy.FIXED),
    chunk_size: int = Form(800, ge=100, le=4000),
    chunk_overlap: int = Form(100, ge=0),
    db: Session = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
) -> Document:
    """Upload a .pdf/.txt, chunk it (fixed | sentence), embed it, store vectors + metadata."""
    data = file.file.read()
    if len(data) > get_settings().max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        return service.ingest(
            db,
            filename=file.filename or "unnamed",
            data=data,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))