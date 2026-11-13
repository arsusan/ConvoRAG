from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Palm Mind RAG API"
    database_url: str = "sqlite:///./data/app.db"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"
    chat_history_turns: int = 6
    chat_ttl_seconds: int = 3600

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    top_k: int = 4
    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()