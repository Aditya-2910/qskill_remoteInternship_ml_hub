from __future__ import annotations

import httpx
import pytest
from conftest import make_settings
from fake_ollama import make_client

from mlhub.benchmark import format_table, parse_llm_label, run_benchmark
from mlhub.llm.agent import ToolRunner
from mlhub.llm.ollama_client import LLMUnavailable, OllamaClient
from mlhub.services import ModelService


def _client(handler) -> OllamaClient:
    return OllamaClient("http://x", "llama3.2", transport=httpx.MockTransport(handler))


def test_unreachable_server_gives_helpful_error():
    def refuse(request):
        raise httpx.ConnectError("refused")

    c = _client(refuse)
    assert c.is_available() is False
    with pytest.raises(LLMUnavailable, match="ollama serve"):
        c.chat([{"role": "user", "content": "hi"}])


def test_missing_model_gives_pull_hint():
    c = OllamaClient("http://fake", "missing-model", transport=make_client()._http._transport)
    with pytest.raises(LLMUnavailable, match="ollama pull missing-model"):
        c.chat([{"role": "user", "content": "hi"}])


def test_server_error_and_malformed_response():
    with pytest.raises(LLMUnavailable, match="HTTP 500"):
        _client(lambda r: httpx.Response(500, text="boom")).chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMUnavailable, match="Unexpected response"):
        _client(lambda r: httpx.Response(200, json={"nope": 1})).chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMUnavailable, match="no embeddings"):
        _client(lambda r: httpx.Response(200, json={"embeddings": []})).embed(["a"])


def test_has_model_understands_latest_tag():
    c = make_client()
    assert c.has_model("llama3.2") and c.has_model("llama3.2:latest")
    assert not c.has_model("llama3.2:1b") and not c.has_model("mistral")


def test_request_payload_contains_tools_and_options():
    seen = {}

    def spy(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    tools = [{"type": "function", "function": {"name": "t"}}]
    assert _client(spy).chat([{"role": "user", "content": "hi"}], tools=tools, temperature=0)["content"] == "ok"
    assert seen["stream"] is False and seen["tools"] == tools and seen["options"] == {"temperature": 0}


# ---------------------------------------------------------------------------- tool runner


def test_tool_runner_validation(fresh_settings):
    runner = ToolRunner(ModelService(fresh_settings))
    assert "petal_width" in runner.run("predict_iris", {"sepal_length": 5})["error"]
    assert "error" in runner.run(
        "predict_iris", {"sepal_length": "tall", "sepal_width": 1, "petal_length": 1, "petal_width": 1}
    )
    assert "error" in runner.run("predict_spam", {"text": "   "})
    assert "Unknown tool" in runner.run("delete_everything", {})["error"]
    # out-of-range latitude is rejected by the same validation the REST API uses
    assert "error" in runner.run("estimate_house_price", {"Latitude": 80})
    # numbers sent as strings are accepted
    ok = runner.run(
        "predict_iris", {"sepal_length": "5.1", "sepal_width": "3.5", "petal_length": "1.4", "petal_width": "0.2"}
    )
    assert ok["species"] == "setosa"


def test_tool_runner_reports_untrained_models(tmp_path):
    runner = ToolRunner(ModelService(make_settings(tmp_path)))
    assert "make train" in runner.run("predict_spam", {"text": "hello"})["error"]


# ---------------------------------------------------------------------------- benchmark


@pytest.mark.parametrize(
    "text,expected",
    [
        ("spam", 1),
        ("Spam.", 1),
        ("**Ham**", 0),
        ("ham\n\nThis looks fine", 0),
        ("This message is spam", 1),
        ("This is not spam.", 0),
        ("I cannot tell", None),
        ("", None),
        ("It could be spam or ham", None),
    ],
)
def test_parse_llm_label(text, expected):
    assert parse_llm_label(text) == expected


class KeywordLLM:
    model = "keyword-bot"

    def chat(self, messages, tools=None, temperature=0.1):
        text = messages[-1]["content"].lower()
        hit = any(w in text for w in ("prize", "claim", "urgent", "winner", "free", "loan", "singles", "offer"))
        return {"content": "spam" if hit else "ham"}


def test_benchmark_runs_and_reports(fresh_settings):
    result = run_benchmark(fresh_settings, n=40, llm=KeywordLLM())
    assert result["n_messages"] == 40
    for key in ("classic", "llm"):
        r = result[key]
        assert all(0 <= r[m] <= 1 for m in ("accuracy", "precision", "recall", "f1"))
        assert r["ms_per_message"] >= 0
    assert result["llm"]["model"] == "keyword-bot" and result["llm"]["unparsed_answers"] == 0
    table = format_table(result)
    assert "classic" in table and "llm" in table
