"""Spam benchmark: classic ML (TF-IDF + linear model) vs. a local LLM used zero-shot.

    python -m mlhub.benchmark --n 100 --model llama3.2

Both classifiers see exactly the same held-out messages (the split is the one used in training).
Reported: accuracy, precision, recall, F1, latency per message and time per 1,000 messages. Running
locally costs no API fees, so the price of the LLM is time and hardware, which the latency shows.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from mlhub.config import Settings, get_settings
from mlhub.datasets import load_spam_df
from mlhub.llm.ollama_client import LLMUnavailable, OllamaClient
from mlhub.training.train_spam import split_spam

log = logging.getLogger(__name__)

LLM_PROMPT = (
    "You are an SMS spam filter. Classify the message below.\n"
    "Reply with exactly one word: spam or ham.\n\n"
    "Message: {text}"
)


def parse_llm_label(content: str) -> int | None:
    """Map free-form LLM output to 1 (spam) / 0 (ham); None if it is neither (or ambiguous)."""
    text = re.sub(r"[^a-z' ]+", " ", content.lower()).strip()
    words = text.split()
    if not words:
        return None
    if words[0].startswith("spam"):  # the requested format: a single word
        return 1
    if words[0].startswith("ham"):
        return 0
    if re.search(r"\b(not|isn't|non|no)\s+spam\b", text):  # "this is not spam"
        return 0
    has_spam = bool(re.search(r"\bspam", text))
    has_ham = bool(re.search(r"\bham\b", text))
    if has_spam != has_ham:
        return 1 if has_spam else 0
    return None


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def run_benchmark(settings: Settings, n: int = 100, llm: Any | None = None) -> dict[str, Any]:
    raw, source = load_spam_df(settings)
    _, X_test, _, y_test = split_spam(raw, settings.random_state)
    n = min(n, len(X_test))
    idx = np.random.default_rng(settings.random_state).choice(len(X_test), size=n, replace=False)
    texts = [X_test.iloc[i] for i in idx]
    y_true = [int(y_test.iloc[i]) for i in idx]

    # --- classic model, one message at a time (what the API does) --------------------------
    pipeline = joblib.load(settings.models_dir / "spam" / "model.joblib")
    started = time.perf_counter()
    classic_pred = [int(pipeline.predict_proba([t])[0][1] >= 0.5) for t in texts]
    classic_total = time.perf_counter() - started

    # --- LLM, zero-shot ------------------------------------------------------------------------
    llm = llm or OllamaClient(
        settings.ollama_host, settings.ollama_model, settings.ollama_embed_model, settings.ollama_timeout
    )
    llm_pred: list[int] = []
    unparsed = 0
    started = time.perf_counter()
    for i, text in enumerate(texts, 1):
        reply = llm.chat([{"role": "user", "content": LLM_PROMPT.format(text=text)}], temperature=0.0)
        label = parse_llm_label(reply.get("content", ""))
        if label is None:
            unparsed += 1
            label = 0  # count an unusable answer as "ham" (the filter let the message through)
        llm_pred.append(label)
        if i % 10 == 0:
            log.info("  LLM %d/%d", i, n)
    llm_total = time.perf_counter() - started

    def timing(total: float) -> dict[str, float]:
        return {
            "ms_per_message": round(1000 * total / n, 3),
            "seconds_per_1000_messages": round(1000 * total / n, 2),
        }

    metadata = json.loads((settings.models_dir / "spam" / "metadata.json").read_text())

    result = {
        "dataset": source,
        "n_messages": n,
        "spam_fraction": round(float(np.mean(y_true)), 3),
        "classic": {
            "model": f"TF-IDF + {metadata.get('algorithm', 'model')}",
            **_metrics(y_true, classic_pred),
            **timing(classic_total),
        },
        "llm": {
            "model": getattr(llm, "model", "unknown"),
            **_metrics(y_true, llm_pred),
            **timing(llm_total),
            "unparsed_answers": unparsed,
        },
    }
    return result


def format_table(result: dict[str, Any]) -> str:
    rows = [("Approach", "Accuracy", "Precision", "Recall", "F1", "ms/message")]
    for key in ("classic", "llm"):
        r = result[key]
        rows.append(
            (
                f"{key} ({r['model']})",
                f"{r['accuracy']:.3f}",
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['f1']:.3f}",
                f"{r['ms_per_message']:.1f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="number of test messages (default 100)")
    parser.add_argument("--model", default=None, help="Ollama model (default: OLLAMA_MODEL)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    settings.ensure_dirs()
    llm = OllamaClient(
        settings.ollama_host, args.model or settings.ollama_model, settings.ollama_embed_model, settings.ollama_timeout
    )
    try:
        llm.list_models()
    except LLMUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    result = run_benchmark(settings, args.n, llm)
    out = settings.reports_dir / "spam_benchmark.json"
    out.write_text(json.dumps(result, indent=2))
    print(format_table(result))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
