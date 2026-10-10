"""A scripted stand-in for Ollama, so tests run without a GPU, a model download or a network.

It speaks the same HTTP API subset the app uses (/api/tags, /api/chat, /api/embed). The "LLM" is a
handful of regexes that decide whether to emit a tool call, which is enough to test the real
agent loop, the real tool execution and the real RAG prompt building.

Two ways to use it:
    * in-process:  ``make_client()`` returns an ``OllamaClient`` wired to ``httpx.MockTransport``
    * over HTTP:   ``python tests/fake_ollama.py`` starts it on port 11435 (used for end-to-end runs)
"""

from __future__ import annotations

import json
import re
import zlib
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

from mlhub.llm.ollama_client import OllamaClient

DIM = 256
NUM = r"-?\d+(?:\.\d+)?"


def _embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(word.encode()) % DIM] += 1.0
    return vec


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


def fake_chat(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload["messages"]
    tools = payload.get("tools")
    last = messages[-1]
    asked = next(m["content"] for m in reversed(messages) if m["role"] == "user").lower()

    if tools and "loop forever" in asked:  # a model that never stops calling tools
        return {"role": "assistant", "content": "", "tool_calls": [_tool_call("predict_spam", {"text": "again"})]}

    if last["role"] == "tool":  # second step: "explain" the tool result
        return {"role": "assistant", "content": f"TOOL RESULT: {last['content']}"}

    user = last["content"]
    lowered = user.lower()

    if tools:
        if "iris" in lowered:
            nums = [float(x) for x in re.findall(NUM, user)]
            if len(nums) >= 4:
                keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call("predict_iris", dict(zip(keys, nums[:4], strict=True)))],
                }
            return {"role": "assistant", "content": "Please give me all four measurements."}
        quoted = re.search(r'"([^"]+)"', user)
        if "spam" in lowered and quoted:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("predict_spam", {"text": quoted.group(1)})],
            }
        income = re.search(rf"income\s*(?:of|is)?\s*({NUM})", lowered)
        if "house" in lowered and income:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("estimate_house_price", {"MedInc": float(income.group(1))})],
            }
        if "bad tool" in lowered:  # a model that hallucinates arguments
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("predict_iris", {"sepal_length": "tall"})],
            }

    # plain answer: prove that RAG context reached the prompt
    system = messages[0]["content"]
    match = re.search(r"\[1\] \(([^)]+)\)", system)
    if match:
        return {"role": "assistant", "content": f"KNOWLEDGE: answered using {match.group(1)}"}
    return {"role": "assistant", "content": "KNOWLEDGE: nothing relevant found"}


def handle_request(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}, {"name": "nomic-embed-text:latest"}]})
    payload = json.loads(request.content or b"{}")
    if path == "/api/chat":
        if payload.get("model") == "missing-model":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json={"model": payload["model"], "message": fake_chat(payload)})
    if path == "/api/embed":
        texts = payload["input"]
        texts = [texts] if isinstance(texts, str) else texts
        return httpx.Response(200, json={"embeddings": [_embed(t) for t in texts]})
    return httpx.Response(404, json={"error": "not found"})


def make_client(model: str = "llama3.2", embed_model: str = "nomic-embed-text") -> OllamaClient:
    return OllamaClient("http://fake-ollama", model, embed_model, transport=httpx.MockTransport(handle_request))


def build_app():
    """The same fake, as a real HTTP server."""
    app = FastAPI(title="fake ollama")

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def _all(path: str, request: Request) -> Response:
        body = await request.body()
        fake_request = httpx.Request(request.method, f"http://fake/{path}", content=body)
        resp = handle_request(fake_request)
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=11435, log_level="warning")
