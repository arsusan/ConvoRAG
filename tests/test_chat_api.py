import json

from fastapi.testclient import TestClient

from tests.conftest import FakeLLM


def _route(intent: str, **slots: str | None) -> str:
    return json.dumps({"intent": intent, "name": None, "email": None, "date": None, "time": None} | slots)


def test_rag_answer_returns_sources(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.responses = [_route("question"), "Interns work 40 hours a week."]
    resp = client.post("/api/v1/chat", json={"session_id": "s1", "message": "How many hours?"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["intent"] == "question"
    assert "40 hours" in body["answer"]
    assert body["sources"][0]["filename"] == "policy.txt"


def test_multi_turn_condenses_followup(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.responses = [_route("question"), "40 hours.",
                          _route("question"), "How many hours do interns work on Fridays?", "Same 40 hours."]
    client.post("/api/v1/chat", json={"session_id": "s2", "message": "Hours per week?"})
    resp = client.post("/api/v1/chat", json={"session_id": "s2", "message": "And on Fridays?"})
    assert resp.json()["answer"] == "Same 40 hours."
    assert fake_llm.responses == []  # router + condense + answer were all consumed


def test_booking_flow_over_two_turns(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.responses = [
        _route("booking", name="Susan Aryal"),
        _route("booking", email="susan@example.com", date="2099-01-15", time="14:30"),
    ]
    first = client.post("/api/v1/chat", json={"session_id": "b1", "message": "Book an interview, I'm Susan Aryal"})
    assert "email address" in first.json()["answer"]
    assert first.json()["booking_id"] is None

    second = client.post("/api/v1/chat", json={"session_id": "b1", "message": "susan@example.com, Jan 15 2099 at 2:30pm"})
    assert second.json()["booking_id"] == 1

    bookings = client.get("/api/v1/bookings").json()
    assert bookings[0]["name"] == "Susan Aryal" and bookings[0]["interview_time"] == "14:30:00"


def test_invalid_email_is_rejected_and_reasked(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.responses = [_route("booking", name="Susan", email="not-an-email", date="2099-01-15", time="10:00")]
    resp = client.post("/api/v1/chat", json={"session_id": "b2", "message": "book it"})
    assert "does not look valid" in resp.json()["answer"]
    assert resp.json()["booking_id"] is None


def test_cancel_clears_state(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.responses = [_route("booking", name="Susan"), _route("cancel_booking")]
    client.post("/api/v1/chat", json={"session_id": "b3", "message": "book, I'm Susan"})
    resp = client.post("/api/v1/chat", json={"session_id": "b3", "message": "never mind"})
    assert resp.json()["intent"] == "cancel_booking"