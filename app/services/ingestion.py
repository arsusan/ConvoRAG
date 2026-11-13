from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.chunking import ChunkingStrategy, chunk_text
from app.services.embeddings import EmbeddingService
from app.services.extractor import ExtractionError, extract_text
from app.services.vector_store import VectorStore

SUPPORTED_TYPES: dict[str, str] = {".pdf": "application/pdf", ".txt": "text/plain"}


class IngestionError(ValueError):
    """Raised for invalid ingestion requests."""


class IngestionService:
    def __init__(self, embedder: EmbeddingService, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def ingest(
        self,
        db: Session,
        *,
        filename: str,
        data: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_TYPES:
            raise IngestionError("Only .pdf and .txt files are supported")
        try:
            text = extract_text(ext, data)
            if not text.strip():
                raise IngestionError("No extractable text found in the file")
            chunks = chunk_text(text, strategy, chunk_size, chunk_overlap)
        except (ExtractionError, ValueError) as exc:
            raise IngestionError(str(exc)) from exc

        vectors = self._embedder.embed_texts(chunks)
        document_id = str(uuid4())
        self._store.upsert_chunks(
            document_id=document_id,
            filename=filename,
            strategy=strategy.value,
            chunks=chunks,
            vectors=vectors,
        )
        document = Document(
            id=document_id,
            filename=filename,
            content_type=SUPPORTED_TYPES[ext],
            size_bytes=len(data),
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_chunks=len(chunks),
        )
        db.add(document)
        db.commit()
        return document