from datetime import date, datetime, time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Intent(str, Enum):
    QUESTION = "question"
    BOOKING = "booking"
    CANCEL_BOOKING = "cancel_booking"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: Intent
    sources: list[SourceChunk] = []
    booking_id: int | None = None


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    name: str
    email: str
    interview_date: date
    interview_time: time
    created_at: datetime