from fastapi.testclient import TestClient

from tests.conftest import FakeVectorStore

SAMPLE = ("Interns work forty hours a week. They get mentorship. " * 40).encode()


def test_upload_txt_fixed(client: TestClient, fake_store: FakeVectorStore) -> None:
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", SAMPLE, "text/plain")},
        data={"strategy": "fixed", "chunk_size": "300", "chunk_overlap": "50"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["num_chunks"] > 1 and body["chunking_strategy"] == "fixed"
    assert len(fake_store.upserts) == 1
    assert client.get("/api/v1/documents").json()[0]["id"] == body["id"]


def test_upload_txt_sentence_strategy(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", SAMPLE, "text/plain")},
        data={"strategy": "sentence"},
    )
    assert resp.status_code == 201
    assert resp.json()["chunking_strategy"] == "sentence"


def test_rejects_unsupported_extension(client: TestClient) -> None:
    resp = client.post("/api/v1/documents", files={"file": ("a.docx", b"x", "application/zip")})
    assert resp.status_code == 400


def test_rejects_empty_file(client: TestClient) -> None:
    resp = client.post("/api/v1/documents", files={"file": ("a.txt", b"   ", "text/plain")})
    assert resp.status_code == 400


def test_rejects_overlap_not_smaller_than_size(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", SAMPLE, "text/plain")},
        data={"chunk_size": "200", "chunk_overlap": "200"},
    )
    assert resp.status_code == 400