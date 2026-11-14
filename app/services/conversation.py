import asyncio
import json
from datetime import date

from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.schemas.chat import ChatMessage, ChatResponse, Intent, SourceChunk
from app.services.booking import BookingSlots, create_booking, validate_slots
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMClient
from app.services.memory import ChatMemory
from app.services.vector_store import RetrievedChunk, VectorStore

_SLOT_LABELS = {"name": "full name", "email": "email address", "date": "preferred date", "time": "preferred time"}

_ROUTER_PROMPT = """You are the routing module of an assistant that answers questions about \
uploaded documents and books interviews. Today's date is {today}.
Return ONLY a JSON object with this exact shape:
{{"intent": "question" | "booking" | "cancel_booking", "name": string|null, "email": string|null, \
"date": "YYYY-MM-DD"|null, "time": "HH:MM"|null}}
Rules:
- intent="booking" if the user wants to schedule/book an interview, OR a booking is in progress \
and the user is supplying booking details.
- intent="cancel_booking" if the user abandons the booking in progress.
- Otherwise intent="question".
- Extract only values the user actually stated in the conversation. Convert relative dates \
("tomorrow", "next Monday") to YYYY-MM-DD and times to 24-hour HH:MM. Use null when unknown.
Booking details collected so far: {state}"""

_CONDENSE_PROMPT = (
    "Rewrite the user's latest message as a standalone question, resolving pronouns and "
    "references using the chat history. Return only the rewritten question."
)

_ANSWER_PROMPT = """You are a helpful assistant. Answer using ONLY the context below. \
If the answer is not in the context, say you could not find it in the documents. \
Be concise.

Context:
{context}"""


class RouterDecision(BaseModel):
    intent: Intent = Intent.QUESTION
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None

    @field_validator("name", "email", "date", "time", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if value is None:
            return None
        text = str(value).strip()
        return None if text.lower() in {"", "null", "none", "n/a"} else text


class ConversationService:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: VectorStore,
        llm: LLMClient,
        memory: ChatMemory,
        top_k: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._memory = memory
        self._top_k = top_k

    async def handle(self, db: Session, session_id: str, message: str) -> ChatResponse:
        history = await self._memory.get_history(session_id)
        state = await self._memory.get_booking_state(session_id)
        decision = await self._route(message, history, state)

        sources: list[SourceChunk] = []
        booking_id: int | None = None

        if decision.intent is Intent.CANCEL_BOOKING:
            await self._memory.clear_booking_state(session_id)
            answer = "No problem, I have cancelled the interview booking. How else can I help?"
        elif decision.intent is Intent.BOOKING:
            answer, booking_id = await self._handle_booking(db, session_id, state, decision)
        else:
            answer, sources = await self._handle_question(message, history)

        await self._memory.append(
            session_id,
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=answer),
        )
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            intent=decision.intent,
            sources=sources,
            booking_id=booking_id,
        )

    # ---------- routing ----------
    async def _route(
        self, message: str, history: list[ChatMessage], state: BookingSlots
    ) -> RouterDecision:
        system = ChatMessage(
            role="system",
            content=_ROUTER_PROMPT.format(today=date.today().isoformat(), state=state.model_dump_json()),
        )
        raw = await self._llm.complete(
            [system, *history, ChatMessage(role="user", content=message)],
            json_mode=True,
            temperature=0.0,
        )
        try:
            return RouterDecision.model_validate(json.loads(raw))
        except (ValueError, TypeError):
            return RouterDecision()  # safe default: treat as a normal question

    # ---------- booking ----------
    async def _handle_booking(
        self, db: Session, session_id: str, state: BookingSlots, decision: RouterDecision
    ) -> tuple[str, int | None]:
        incoming = BookingSlots(
            name=decision.name, email=decision.email, date=decision.date, time=decision.time
        )
        cleaned, errors = validate_slots(state.merge(incoming))
        missing = cleaned.missing()

        if missing:
            await self._memory.set_booking_state(session_id, cleaned)
            needed = ", ".join(_SLOT_LABELS[m] for m in missing)
            prefix = (" ".join(errors) + " ") if errors else ""
            return f"{prefix}To book your interview, I still need your {needed}.", None

        booking = create_booking(db, session_id, cleaned)
        await self._memory.clear_booking_state(session_id)
        message = (
            f"Your interview is booked, {booking.name}. Date: {booking.interview_date}, "
            f"time: {booking.interview_time.strftime('%H:%M')}, email: {booking.email}. "
            f"Confirmation ID: {booking.id}."
        )
        return message, booking.id

    # ---------- custom RAG ----------
    async def _handle_question(
        self, message: str, history: list[ChatMessage]
    ) -> tuple[str, list[SourceChunk]]:
        standalone = await self._standalone_question(message, history)
        vector = await asyncio.to_thread(self._embedder.embed_query, standalone)
        chunks = await asyncio.to_thread(self._store.search, vector, self._top_k)
        if not chunks:
            return "I could not find anything relevant in the ingested documents.", []

        system = ChatMessage(role="system", content=_ANSWER_PROMPT.format(context=self._format(chunks)))
        answer = await self._llm.complete([system, *history, ChatMessage(role="user", content=message)])
        sources = [
            SourceChunk(
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                score=round(c.score, 4),
                text=c.text[:300],
            )
            for c in chunks
        ]
        return answer.strip(), sources

    async def _standalone_question(self, message: str, history: list[ChatMessage]) -> str:
        if not history:
            return message
        rewritten = await self._llm.complete(
            [
                ChatMessage(role="system", content=_CONDENSE_PROMPT),
                *history,
                ChatMessage(role="user", content=message),
            ],
            temperature=0.0,
        )
        return rewritten.strip() or message

    @staticmethod
    def _format(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(f"[{i}] ({c.filename}) {c.text}" for i, c in enumerate(chunks, 1))