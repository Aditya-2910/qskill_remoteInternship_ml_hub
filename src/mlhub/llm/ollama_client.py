"""Minimal Ollama HTTP client (chat with tool calling + embeddings).

Only depends on ``httpx``. Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Ollama is not reachable, or the requested model is not pulled."""


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self._http = httpx.Client(base_url=self.host, timeout=timeout, transport=transport)

    # ------------------------------------------------------------------ helpers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._http.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.host} ({type(exc).__name__}). "
                "Is it running? Start it with `ollama serve`."
            ) from exc
        if resp.status_code == 404:
            model = payload.get("model")
            raise LLMUnavailable(f"Ollama does not have the model '{model}'. Run `ollama pull {model}`.")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    # --------------------------------------------------------------------- API

    def list_models(self, timeout: float = 5.0) -> list[str]:
        try:
            resp = self._http.get("/api/tags", timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Cannot reach Ollama at {self.host}: {exc}") from exc
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except LLMUnavailable:
            return False

    def has_model(self, name: str | None = None) -> bool:
        wanted = name or self.model
        try:
            installed = self.list_models()
        except LLMUnavailable:
            return False
        if ":" in wanted:
            return wanted in installed
        # "llama3.2" means "llama3.2:latest" to Ollama
        return any(m.split(":")[0] == wanted for m in installed)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """One non-streaming chat turn. Returns the assistant ``message`` dict."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        data = self._post("/api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMUnavailable(f"Unexpected response from Ollama: {str(data)[:300]}")
        return message

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": self.embed_model, "input": texts})
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise LLMUnavailable("Ollama returned no embeddings")
        return embeddings

    def close(self) -> None:
        self._http.close()
