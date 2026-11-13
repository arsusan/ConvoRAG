import re
from datetime import date, datetime

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Booking

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class BookingSlots(BaseModel):
    name: str | None = None
    email: str | None = None
    date: str | None = None  # YYYY-MM-DD
    time: str | None = None  # HH:MM (24h)

    def missing(self) -> list[str]:
        return [f for f in ("name", "email", "date", "time") if not getattr(self, f)]

    def merge(self, other: "BookingSlots") -> "BookingSlots":
        data = self.model_dump()
        data.update({k: v for k, v in other.model_dump().items() if v})
        return BookingSlots(**data)

    def is_empty(self) -> bool:
        return not any(self.model_dump().values())


def validate_slots(slots: BookingSlots, today: date | None = None) -> tuple[BookingSlots, list[str]]:
    """Drop invalid values and return human-readable errors for them."""
    today = today or date.today()
    data = slots.model_dump()
    errors: list[str] = []

    if data["name"] and len(data["name"].strip()) < 2:
        data["name"] = None
        errors.append("The name looks too short.")
    if data["email"] and not _EMAIL_RE.match(data["email"].strip()):
        data["email"] = None
        errors.append("That email address does not look valid.")
    if data["date"]:
        try:
            if date.fromisoformat(data["date"]) < today:
                raise ValueError
        except ValueError:
            data["date"] = None
            errors.append("The date must be a valid date that is not in the past.")
    if data["time"]:
        try:
            datetime.strptime(data["time"], "%H:%M")
        except ValueError:
            data["time"] = None
            errors.append("The time must be a valid time such as 14:30.")
    return BookingSlots(**data), errors


def create_booking(db: Session, session_id: str, slots: BookingSlots) -> Booking:
    assert slots.name and slots.email and slots.date and slots.time
    booking = Booking(
        session_id=session_id,
        name=slots.name.strip(),
        email=slots.email.strip(),
        interview_date=date.fromisoformat(slots.date),
        interview_time=datetime.strptime(slots.time, "%H:%M").time(),
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    return booking