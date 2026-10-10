"""Streamlit front end for ML Hub. It only talks to the FastAPI backend over HTTP.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("MLHUB_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("MLHUB_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="ML Hub", page_icon="🧠", layout="wide")


# ------------------------------------------------------------------------------ API helpers


def call(method: str, path: str, timeout: float = 30, **kwargs: Any) -> tuple[httpx.Response | None, str | None]:
    """Returns (response, None) on success or (None, human readable error)."""
    try:
        resp = httpx.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Cannot reach the API at {API_URL} ({type(exc).__name__}). Start it with `make api`."
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(f"{'.'.join(map(str, d['loc'][1:]))}: {d['msg']}" for d in detail)
        return None, f"API error {resp.status_code}: {detail}"
    return resp, None


def get_json(path: str, **kw: Any) -> tuple[Any, str | None]:
    resp, err = call("GET", path, **kw)
    return (resp.json() if resp is not None else None), err


def post_json(path: str, payload: dict[str, Any], **kw: Any) -> tuple[Any, str | None]:
    resp, err = call("POST", path, json=payload, **kw)
    return (resp.json() if resp is not None else None), err


@st.cache_data(ttl=30, show_spinner=False)
def cached_models() -> dict[str, Any]:
    data, _ = get_json("/v1/models")
    return data or {}


@st.cache_data(ttl=300, show_spinner=False)
def cached_plot(model: str, filename: str) -> bytes | None:
    resp, _ = call("GET", f"/v1/models/{model}/plots/{filename}")
    return resp.content if resp is not None else None


# ------------------------------------------------------------------------------- sidebar

st.sidebar.title("🧠 ML Hub")
PAGES = ["Assistant", "Iris", "Spam", "House prices", "Models", "Monitoring"]
page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")

health, health_err = get_json("/health", timeout=8)
if health_err:
    st.sidebar.error("API offline")
    st.error(health_err)
    st.stop()

st.sidebar.success(f"API {health['version']} · {health['status']}")
for name, ready in health["models"].items():
    st.sidebar.write(("✅" if ready else "❌") + f" {name} model")
ollama = health["ollama"]
if not ollama["reachable"]:
    st.sidebar.warning("Ollama not reachable: the Assistant is disabled. Start it with `ollama serve`.")
elif not ollama["model_available"]:
    st.sidebar.warning(f"Ollama model missing. Run `ollama pull {ollama['model']}`.")
else:
    st.sidebar.write(f"✅ LLM `{ollama['model']}`")
st.sidebar.markdown(f"[API docs]({API_URL}/docs)")
if not all(health["models"].values()):
    st.warning("Some models are not trained yet. Run `make train` and restart the API.")


# ---------------------------------------------------------------------------------- pages


def page_assistant() -> None:
    st.header("Assistant")
    st.caption(
        "A local LLM (via Ollama) that calls the trained models as tools and explains results "
        "using a small knowledge base (RAG)."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    examples = [
        'Is this message spam? "URGENT! You won a free prize, call 0900123456 to claim"',
        "Classify an iris with sepal 5.9 x 3.0 cm and petal 5.1 x 1.8 cm",
        "Estimate the price of a house in a block with median income 6.5 near San Francisco (37.8, -122.4)",
        "Why is precision important for a spam filter?",
    ]
    cols = st.columns(len(examples))
    clicked = None
    for col, text in zip(cols, examples, strict=True):
        if col.button(text[:48] + "…", help=text, key=f"ex-{text[:20]}"):
            clicked = text

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for tc in msg.get("tool_calls", []):
                with st.expander(f"🔧 Tool call: {tc['name']}"):
                    st.json({"arguments": tc["arguments"], "result": tc["result"]})
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} knowledge-base sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}** (score {s['score']})")
                        st.caption(s["text"])

    prompt = st.chat_input("Ask about the models, or ask me to run one…") or clicked
    if not prompt:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-10:]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            data, err = post_json("/v1/chat", {"message": prompt, "history": history}, timeout=300)
        if err:
            st.error(err)
            st.session_state.messages.pop()  # do not keep a question that got no answer
            return
        st.markdown(data["answer"])
    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "tool_calls": data["tool_calls"], "sources": data["sources"]}
    )
    st.rerun()


IRIS_PRESETS = {
    "setosa": (5.0, 3.4, 1.5, 0.2),
    "versicolor": (5.9, 2.8, 4.3, 1.3),
    "virginica": (6.6, 3.0, 5.6, 2.0),
}
IRIS_FIELDS = [
    ("sepal_length", "Sepal length (cm)", 4.0, 8.0),
    ("sepal_width", "Sepal width (cm)", 2.0, 4.5),
    ("petal_length", "Petal length (cm)", 1.0, 7.0),
    ("petal_width", "Petal width (cm)", 0.1, 2.5),
]


def _set_iris_preset(species: str) -> None:
    for (key, *_), value in zip(IRIS_FIELDS, IRIS_PRESETS[species], strict=True):
        st.session_state[f"iris_{key}"] = value


def page_iris() -> None:
    from sklearn.datasets import load_iris

    st.header("Iris flower classifier")
    st.caption("Task 1: classify a flower into setosa, versicolor or virginica from its measurements.")
    st.write("Presets:")
    cols = st.columns(len(IRIS_PRESETS))
    for col, species in zip(cols, IRIS_PRESETS, strict=True):
        col.button(f"Typical {species}", on_click=_set_iris_preset, args=(species,), key=f"preset-{species}")

    values = {}
    left, right = st.columns([1, 2])
    with left:
        for i, (key, label, lo, hi) in enumerate(IRIS_FIELDS):
            st.session_state.setdefault(f"iris_{key}", IRIS_PRESETS["versicolor"][i])
            values[key] = st.slider(label, lo, hi, step=0.1, key=f"iris_{key}")
        run = st.button("Classify", type="primary")

    with right:
        data = load_iris(as_frame=True).frame
        data["species"] = data["target"].map(dict(enumerate(load_iris().target_names)))
        plot = data[["petal length (cm)", "petal width (cm)", "species"]].rename(
            columns={"petal length (cm)": "petal_length", "petal width (cm)": "petal_width"}
        )
        plot["size"] = 30
        plot = pd.concat(
            [
                plot,
                pd.DataFrame(
                    [
                        {
                            "petal_length": values["petal_length"],
                            "petal_width": values["petal_width"],
                            "species": "your flower",
                            "size": 300,
                        }
                    ]
                ),
            ]
        )
        st.scatter_chart(plot, x="petal_length", y="petal_width", color="species", size="size")

    if run:
        pred, err = post_json("/v1/predict/iris", values)
        if err:
            st.error(err)
            return
        st.success(f"Predicted species: **{pred['prediction']}**")
        st.bar_chart(pd.Series(pred["probabilities"], name="probability"))
        st.caption(f"model: {pred['algorithm']} · version {pred['model_version']} · {pred['latency_ms']} ms")


SPAM_EXAMPLES = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now to claim",
    "Friendly message": "Hey, are we still on for lunch at 1pm? I'll be a few minutes late, save me a seat",
    "Account phishing": "URGENT! Your account has been suspended. Verify your details at http://bit.ly/x9 in 24 hours",
}


def page_spam() -> None:
    st.header("Spam detector")
    st.caption("Task 2: TF-IDF + a linear classifier. Shows which words pushed the decision.")
    for label, text in SPAM_EXAMPLES.items():
        st.button(label, key=f"spam-ex-{label}", on_click=lambda t=text: st.session_state.update(spam_text=t))
    text = st.text_area("Message", key="spam_text", height=120, placeholder="Paste a message…")
    if st.button("Check message", type="primary"):
        if not text.strip():
            st.warning("Type or paste a message first.")
            return
        pred, err = post_json("/v1/predict/spam", {"text": text})
        if err:
            st.error(err)
            return
        spam_p = pred["probabilities"]["spam"]
        (st.error if pred["prediction"] == "spam" else st.success)(
            f"**{pred['prediction'].upper()}** · spam probability {spam_p:.1%}"
        )
        st.progress(min(max(spam_p, 0.0), 1.0))
        terms = pred["details"].get("top_terms", [])
        if terms:
            st.write("Most influential words:")
            st.dataframe(
                pd.DataFrame(terms).rename(columns={"term": "word", "push": "pushes towards"}), hide_index=True
            )
        st.caption(f"model: {pred['algorithm']} · {pred['latency_ms']} ms · the message text is not stored")


HOUSE_FIELDS = [
    ("MedInc", "Median income ($10k)", 0.5, 15.0, 0.1),
    ("HouseAge", "House age (years)", 1.0, 52.0, 1.0),
    ("AveRooms", "Avg rooms / household", 1.0, 12.0, 0.1),
    ("AveBedrms", "Avg bedrooms / household", 0.5, 4.0, 0.05),
    ("Population", "Population", 3.0, 12000.0, 10.0),
    ("AveOccup", "Avg occupants / household", 1.0, 8.0, 0.1),
    ("Latitude", "Latitude", 32.5, 42.0, 0.01),
    ("Longitude", "Longitude", -124.4, -114.3, 0.01),
]


def page_house() -> None:
    st.header("House price estimator")
    st.caption("Task 3: regression on the California Housing data (values are 1990 block-group medians).")
    defaults = cached_models().get("house", {}).get("feature_defaults", {})
    values: dict[str, float] = {}
    cols = st.columns(4)
    for i, (key, label, lo, hi, step) in enumerate(HOUSE_FIELDS):
        default = float(min(max(defaults.get(key, (lo + hi) / 2), lo), hi))
        values[key] = cols[i % 4].number_input(label, lo, hi, default, step, key=f"house_{key}")
    if st.button("Estimate price", type="primary"):
        pred, err = post_json("/v1/predict/house", values)
        if err:
            st.error(err)
            return
        st.metric("Estimated median house value", f"${pred['prediction']:,.0f}")
        st.map(pd.DataFrame({"lat": [values["Latitude"]], "lon": [values["Longitude"]]}), zoom=5)
        st.caption(f"model: {pred['algorithm']} · version {pred['model_version']} · {pred['latency_ms']} ms")


def page_models() -> None:
    st.header("Models")
    models = cached_models()
    if not models:
        st.info("No trained models found. Run `make train`.")
        return
    for name, meta in models.items():
        with st.expander(f"{name}: {meta['algorithm']} ({meta['task']})", expanded=True):
            st.caption(
                f"version {meta['version']} · trained {meta['trained_at']} · data: {meta['data_source']} · "
                f"{meta['n_train']} train / {meta['n_test']} test"
            )
            if "synthetic" in meta["data_source"]:
                st.warning("Trained on a synthetic offline stand-in dataset, not the real data.")
            metric_cols = st.columns(len(meta["test_metrics"]))
            for col, (k, v) in zip(metric_cols, meta["test_metrics"].items(), strict=True):
                col.metric(k, f"{v:.3f}")
            st.write(f"Cross-validation ({meta['cv_metric']}) on the training set, used to pick the model:")
            cv = pd.DataFrame(
                [
                    {"candidate": n, "cv score": round(v, 4), "selected": "✅" if n == meta["algorithm"] else ""}
                    for n, v in sorted(meta["cv_scores"].items(), key=lambda kv: -kv[1])
                ]
            )
            st.dataframe(cv, hide_index=True)
            plots = st.columns(3)
            for i, fname in enumerate(sorted(_plot_names(name))):
                img = cached_plot(name, fname)
                if img:
                    plots[i % 3].image(img, caption=fname)
    if "mlflow_run_id" in next(iter(models.values())):
        st.caption("Compare all runs in the MLflow UI: `mlflow server --backened-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000` then open http://localhost:5000")


def _plot_names(model: str) -> list[str]:
    known = {
        "iris": ["iris_histograms.png", "iris_scatter.png", "iris_confusion.png"],
        "spam": ["spam_eda.png", "spam_confusion.png"],
        "house": ["house_histograms.png", "house_correlation.png", "house_pred_vs_actual.png"],
    }
    return known.get(model, [])


def page_monitoring() -> None:
    st.header("Monitoring")
    summary, err = get_json("/v1/monitoring/summary")
    if err:
        st.error(err)
        return
    st.metric("Predictions logged", summary["total_predictions"])
    if not summary["models"]:
        st.info("No predictions yet. Try one of the model pages.")
    for name, stats in summary["models"].items():
        st.subheader(name)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions", stats["count"])
        c2.metric("Mean latency", f"{stats['latency_ms_mean']} ms")
        c3.metric("p95 latency", f"{stats['latency_ms_p95']} ms")
        if stats["prediction_counts"]:
            st.bar_chart(pd.Series(stats["prediction_counts"], name="predictions"))
        drift, derr = get_json(f"/v1/monitoring/drift/{name}")
        if derr:
            st.error(derr)
        elif drift["status"] == "not_enough_data":
            st.caption(f"Drift check needs at least {drift['min_samples']} predictions ({drift['n_live']} so far).")
        else:
            (st.error if drift["drift_detected"] else st.success)(
                "Drift detected: consider retraining" if drift["drift_detected"] else "No drift detected"
            )
            st.dataframe(pd.DataFrame(drift["features"]), hide_index=True)


{
    "Assistant": page_assistant,
    "Iris": page_iris,
    "Spam": page_spam,
    "House prices": page_house,
    "Models": page_models,
    "Monitoring": page_monitoring,
}[page]()
