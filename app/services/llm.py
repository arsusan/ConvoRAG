from collections.abc import Sequence

import httpx

from app.schemas.chat import ChatMessage


class LLMError(RuntimeError):
    """Raised when the LLM provider fails."""


class LLMClient:
    """Minimal async client for any OpenAI-compatible chat endpoint (Groq here)."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        if not self._api_key:
            raise LLMError("GROQ_API_KEY is not configured")
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"LLM error {resp.status_code}: {resp.text[:300]}")
        return str(resp.json()["choices"][0]["message"]["content"])