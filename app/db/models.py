from datetime import UTC, date, datetime, time
from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int]
    chunking_strategy: Mapped[str] = mapped_column(String(20))
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    num_chunks: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254))
    interview_date: Mapped[date]
    interview_time: Mapped[time]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)