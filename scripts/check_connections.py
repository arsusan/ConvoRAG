import asyncio

from qdrant_client import QdrantClient
from redis.asyncio import Redis

from app.core.config import get_settings


async def check_redis(url: str) -> None:
    client = Redis.from_url(url, decode_responses=True)
    await client.set("healthcheck", "ok", ex=30)
    print("Redis OK:", await client.get("healthcheck"))
    await client.aclose()


def check_qdrant(url: str, api_key: str | None) -> None:
    client = QdrantClient(url=url, api_key=api_key)
    print("Qdrant OK, collections:", [c.name for c in client.get_collections().collections])


if __name__ == "__main__":
    s = get_settings()
    check_qdrant(s.qdrant_url, s.qdrant_api_key)
    asyncio.run(check_redis(s.redis_url))