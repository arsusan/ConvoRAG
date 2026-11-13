from redis.asyncio import Redis

from app.schemas.chat import ChatMessage
from app.services.booking import BookingSlots


class ChatMemory:
    """Redis-backed chat history and in-progress booking state, per session."""

    def __init__(self, client: Redis, history_turns: int, ttl_seconds: int) -> None:
        self._client = client
        self._max_messages = history_turns * 2
        self._ttl = ttl_seconds

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"chat:{session_id}:history"

    @staticmethod
    def _booking_key(session_id: str) -> str:
        return f"chat:{session_id}:booking"

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        raw = await self._client.lrange(self._history_key(session_id), 0, -1)
        return [ChatMessage.model_validate_json(item) for item in raw]

    async def append(self, session_id: str, *messages: ChatMessage) -> None:
        key = self._history_key(session_id)
        await self._client.rpush(key, *[m.model_dump_json() for m in messages])
        await self._client.ltrim(key, -self._max_messages, -1)
        await self._client.expire(key, self._ttl)

    async def get_booking_state(self, session_id: str) -> BookingSlots:
        raw = await self._client.get(self._booking_key(session_id))
        return BookingSlots.model_validate_json(raw) if raw else BookingSlots()

    async def set_booking_state(self, session_id: str, slots: BookingSlots) -> None:
        await self._client.set(self._booking_key(session_id), slots.model_dump_json(), ex=self._ttl)

    async def clear_booking_state(self, session_id: str) -> None:
        await self._client.delete(self._booking_key(session_id))