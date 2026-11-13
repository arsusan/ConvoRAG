from functools import lru_cache

from fastapi import Depends
from qdrant_client import QdrantClient
from redis.asyncio import Redis

from app.core.config import get_settings
from app.db.session import get_db  # re-exported for routes and test overrides
from app.services.conversation import ConversationService
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.llm import LLMClient
from app.services.memory import ChatMemory
from app.services.vector_store import VectorStore

__all__ = ["get_db"]


@lru_cache
def get_embedder() -> EmbeddingService:
    return EmbeddingService(get_settings().embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    return VectorStore(client, s.qdrant_collection, s.embedding_dim)


@lru_cache
def get_llm() -> LLMClient:
    s = get_settings()
    return LLMClient(s.groq_api_key, s.groq_base_url, s.groq_model)


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def get_memory(redis: Redis = Depends(get_redis)) -> ChatMemory:
    s = get_settings()
    return ChatMemory(redis, s.chat_history_turns, s.chat_ttl_seconds)


def get_ingestion_service(
    embedder: EmbeddingService = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> IngestionService:
    return IngestionService(embedder, store)


def get_conversation_service(
    embedder: EmbeddingService = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(get_llm),
    memory: ChatMemory = Depends(get_memory),
) -> ConversationService:
    return ConversationService(embedder, store, llm, memory, get_settings().top_k)