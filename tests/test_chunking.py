import pytest

from app.services.chunking import fixed_size_chunks, sentence_chunks

TEXT = " ".join(f"This is sentence number {i}." for i in range(40))


def test_fixed_chunks_respect_size_and_overlap() -> None:
    chunks = fixed_size_chunks("a" * 250, chunk_size=100, overlap=20)
    assert [len(c) for c in chunks] == [100, 100, 90]


def test_sentence_chunks_stay_within_limit_and_overlap() -> None:
    chunks = sentence_chunks(TEXT, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    last_sentence = chunks[0].split(". ")[-1].rstrip(".")
    assert last_sentence in chunks[1]  # overlap carried forward


def test_oversized_sentence_falls_back_to_fixed() -> None:
    chunks = sentence_chunks("x" * 500 + ".", chunk_size=200, overlap=20)
    assert len(chunks) >= 3


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        fixed_size_chunks("hello", size, overlap)