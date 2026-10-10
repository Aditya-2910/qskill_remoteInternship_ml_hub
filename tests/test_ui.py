"""Drive the real Streamlit script with Streamlit's AppTest against a real API served over HTTP.

The API runs in a background thread on a free port with the fake Ollama behind it, so this
exercises the exact HTTP path the browser UI uses.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

pytest.importorskip("streamlit")
from fake_ollama import make_client  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from mlhub.api.main import create_app  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture()
def live_api(fresh_settings, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(fresh_settings, llm=make_client()), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    monkeypatch.setenv("MLHUB_API_URL", f"http://127.0.0.1:{port}")
    yield port
    server.should_exit = True
    thread.join(timeout=5)


def _run(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_pages_render(live_api):
    for page in ["Assistant", "Iris", "Spam", "House prices", "Models", "Monitoring"]:
        at = _run(page)
        assert at.header, f"{page} page rendered no header"


def test_iris_classification_flow(live_api):
    at = _run("Iris")
    at.button(key="preset-virginica").click().run()
    next(b for b in at.button if b.label == "Classify").click().run()
    assert not at.exception
    assert any("virginica" in s.value for s in at.success)


def test_spam_flow(live_api):
    at = _run("Spam")
    at.button(key="spam-ex-Prize scam").click().run()
    next(b for b in at.button if b.label == "Check message").click().run()
    assert any("SPAM" in e.value for e in at.error)
    assert at.dataframe, "influential words table should be shown"


def test_house_flow(live_api):
    at = _run("House prices")
    next(b for b in at.button if b.label == "Estimate price").click().run()
    assert not at.exception
    assert at.metric[0].value.startswith("$")


def test_assistant_flow(live_api):
    at = _run("Assistant")
    at.chat_input[0].set_value('Is this spam? "WINNER claim your free prize now"').run()
    assert not at.exception
    texts = [m.value for m in at.markdown]
    assert any("TOOL RESULT" in t for t in texts)
    assert any("Tool call: predict_spam" in e.label for e in at.expander)


def test_monitoring_shows_logged_predictions(live_api):
    at = _run("Spam")
    at.button(key="spam-ex-Prize scam").click().run()
    next(b for b in at.button if b.label == "Check message").click().run()
    at.sidebar.radio[0].set_value("Monitoring").run()
    assert at.metric[0].value == "1"


def test_ui_reports_offline_api(monkeypatch):
    monkeypatch.setenv("MLHUB_API_URL", "http://127.0.0.1:1")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("Cannot reach the API" in e.value for e in at.error)
