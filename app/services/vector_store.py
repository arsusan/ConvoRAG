from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        strategy: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=list(vec),
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "strategy": strategy,
                    "text": text,
                },
            )
            for idx, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), 64):
            self._client.upsert(self._collection, points=points[start : start + 64])
        return len(points)

    def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        result = self._client.query_points(
            self._collection, query=list(vector), limit=top_k, with_payload=True
        )
        return [
            RetrievedChunk(
                document_id=str(p.payload["document_id"]),
                filename=str(p.payload["filename"]),
                chunk_index=int(p.payload["chunk_index"]),
                score=float(p.score),
                text=str(p.payload["text"]),
            )
            for p in result.points
            if p.payload
        ]