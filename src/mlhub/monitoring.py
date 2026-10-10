"""Prediction logging and a lightweight data-drift check.

Every prediction is appended to ``logs/predictions.jsonl``. For spam we log only the message
length and word count (never the text itself). The drift check compares the logged inputs with a
reference sample saved at training time using a two-sample Kolmogorov-Smirnov test. It is a small
stand-in for a tool like Evidently, small enough to read in one sitting.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

MAX_RECORDS = 5000  # only the most recent predictions are considered


class PredictionLogger:
    def __init__(self, logs_dir: Path):
        self.path = Path(logs_dir) / "predictions.jsonl"
        self._lock = threading.Lock()

    def log(self, model: str, inputs: dict[str, Any], prediction: Any, latency_ms: float) -> None:
        record = {
            "ts": time.time(),
            "model": model,
            "inputs": inputs,
            "prediction": prediction if isinstance(prediction, str | int | float) else str(prediction),
            "latency_ms": round(latency_ms, 3),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def records(self, model: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-MAX_RECORDS:]
        out = []
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if model is None or rec.get("model") == model:
                out.append(rec)
        return out

    def summary(self) -> dict[str, Any]:
        recs = self.records()
        result: dict[str, Any] = {"total_predictions": len(recs), "models": {}}
        for name in sorted({r["model"] for r in recs}):
            subset = [r for r in recs if r["model"] == name]
            latencies = np.array([r["latency_ms"] for r in subset])
            result["models"][name] = {
                "count": len(subset),
                "latency_ms_mean": round(float(latencies.mean()), 3),
                "latency_ms_p95": round(float(np.percentile(latencies, 95)), 3),
                "prediction_counts": dict(Counter(str(r["prediction"]) for r in subset) if name != "house" else {}),
                "last_prediction_ts": subset[-1]["ts"],
            }
        return result

    def drift_report(
        self, model: str, reference_csv: Path, min_samples: int = 20, alpha: float = 0.01
    ) -> dict[str, Any]:
        recs = self.records(model)
        report: dict[str, Any] = {
            "model": model,
            "n_live": len(recs),
            "min_samples": min_samples,
            "features": [],
            "drift_detected": False,
        }
        if len(recs) < min_samples or not Path(reference_csv).exists():
            report["status"] = "not_enough_data"
            return report
        reference = pd.read_csv(reference_csv)
        live = pd.DataFrame([r["inputs"] for r in recs])
        for col in reference.columns:
            if col not in live.columns:
                continue
            stat, p_value = ks_2samp(reference[col].dropna(), live[col].dropna())
            drifted = bool(p_value < alpha)
            report["features"].append(
                {
                    "feature": col,
                    "ks_statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 6),
                    "drift": drifted,
                }
            )
            report["drift_detected"] |= drifted
        report["status"] = "ok"
        return report
