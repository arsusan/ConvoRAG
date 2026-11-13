from collections.abc import Iterator

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import deps
from app.db.models import Base
from app.main import app
from app.services.memory import ChatMemory
from app.services.vector_store import RetrievedChunk


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []
        self.results: list[RetrievedChunk] = [
            RetrievedChunk("doc-1", "policy.txt", 0, 0.91, "Interns work 40 hours a week.")
        ]

    def upsert_chunks(self, **kwargs: object) -> int:
        self.upserts.append(kwargs)
        return len(kwargs["chunks"])  # type: ignore[arg-type]

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        return self.results[:top_k]


class FakeLLM:
    """Returns scripted responses in order."""

    def __init__(self) -> None:
        self.responses: list[str] = []

    async def complete(self, messages: object, **kwargs: object) -> str:
        return self.responses.pop(0)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def client(fake_llm: FakeLLM, fake_store: FakeVectorStore) -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_db() -> Iterator[Session]:
        with factory() as session:
            yield session

    memory = ChatMemory(fakeredis.aioredis.FakeRedis(decode_responses=True), 6, 3600)

    app.dependency_overrides[deps.get_db] = override_db
    app.dependency_overrides[deps.get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[deps.get_vector_store] = lambda: fake_store
    app.dependency_overrides[deps.get_llm] = lambda: fake_llm
    app.dependency_overrides[deps.get_memory] = lambda: memory
    yield TestClient(app)  # no `with`: skips lifespan (no real Qdrant needed)
    app.dependency_overrides.clear()