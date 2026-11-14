import re
from enum import StrEnum


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"        # sliding character window
    SENTENCE = "sentence"  # sentence-aware packing


_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Strategy 1: fixed character windows with overlap."""
    _validate(chunk_size, overlap)
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):  # window reached the end; no redundant tail chunk
            break
        start += step
    return chunks


def sentence_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Strategy 2: pack whole sentences up to chunk_size, carrying trailing
    sentences (up to `overlap` characters) into the next chunk."""
    _validate(chunk_size, overlap)
    sentences = [s.strip() for s in _BOUNDARY.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        if len(sentence) > chunk_size:  # oversize sentence: fall back to fixed windows
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            chunks.extend(fixed_size_chunks(sentence, chunk_size, overlap))
            continue

        if current and length + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            current, length = carry, carry_len

        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str, strategy: ChunkingStrategy, chunk_size: int, overlap: int) -> list[str]:
    if strategy is ChunkingStrategy.FIXED:
        return fixed_size_chunks(text, chunk_size, overlap)
    return sentence_chunks(text, chunk_size, overlap)