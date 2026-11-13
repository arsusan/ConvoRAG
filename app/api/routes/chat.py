from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_conversation_service, get_db
from app.db.models import Booking
from app.schemas.chat import BookingResponse, ChatRequest, ChatResponse
from app.services.conversation import ConversationService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    service: ConversationService = Depends(get_conversation_service),
) -> ChatResponse:
    """Multi-turn conversational RAG with interview booking."""
    return await service.handle(db, request.session_id, request.message)


@router.get("/bookings", response_model=list[BookingResponse])
def list_bookings(db: Session = Depends(get_db)) -> list[Booking]:
    return list(db.scalars(select(Booking).order_by(Booking.id.desc())))