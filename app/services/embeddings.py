from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:  # lazy load: keeps startup and tests fast
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]