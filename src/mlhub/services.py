"""Model registry + prediction services used by the API and by the chat agent's tools."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mlhub.config import Settings
from mlhub.datasets import HOUSE_FEATURES, IRIS_FEATURES
from mlhub.monitoring import PredictionLogger
from mlhub.schemas import HouseFeatures, IrisFeatures, Prediction

MODEL_NAMES = ("iris", "spam", "house")


class ModelNotAvailable(RuntimeError):
    """Raised when a model has not been trained yet."""


@dataclass
class LoadedModel:
    name: str
    pipeline: Any
    metadata: dict[str, Any]
    directory: Path


class ModelService:
    """Loads models lazily from ``models/<name>/`` and turns raw inputs into predictions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = PredictionLogger(settings.logs_dir)
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ registry

    def get(self, name: str) -> LoadedModel:
        with self._lock:
            if name not in self._models:
                directory = self.settings.models_dir / name
                model_path = directory / "model.joblib"
                if not model_path.exists():
                    raise ModelNotAvailable(
                        f"Model '{name}' has not been trained yet. Run `make train` "
                        f"(python -m mlhub.training.train_all) first."
                    )
                metadata = json.loads((directory / "metadata.json").read_text())
                self._models[name] = LoadedModel(name, joblib.load(model_path), metadata, directory)
            return self._models[name]

    def reload(self) -> None:
        with self._lock:
            self._models.clear()

    def available(self) -> dict[str, bool]:
        return {n: (self.settings.models_dir / n / "model.joblib").exists() for n in MODEL_NAMES}

    def metadata(self, name: str) -> dict[str, Any]:
        return self.get(name).metadata

    def all_metadata(self) -> dict[str, dict[str, Any]]:
        return {n: self.get(n).metadata for n, ok in self.available().items() if ok}

    def _finish(
        self,
        model: LoadedModel,
        started: float,
        prediction: Any,
        probabilities: dict[str, float] | None,
        logged_inputs: dict[str, Any],
        details: dict[str, Any] | None = None,
    ) -> Prediction:
        latency_ms = (time.perf_counter() - started) * 1000
        self.logger.log(model.name, logged_inputs, prediction, latency_ms)
        return Prediction(
            model=model.name,
            model_version=str(model.metadata.get("version", "unknown")),
            algorithm=str(model.metadata.get("algorithm", "unknown")),
            prediction=prediction,
            probabilities=probabilities,
            latency_ms=round(latency_ms, 3),
            details=details or {},
        )

    # ---------------------------------------------------------------- iris

    def predict_iris(self, features: IrisFeatures) -> Prediction:
        model = self.get("iris")
        started = time.perf_counter()
        row = pd.DataFrame([features.model_dump()])[IRIS_FEATURES]
        species = str(model.pipeline.predict(row)[0])
        proba = model.pipeline.predict_proba(row)[0]
        probabilities = {str(c): round(float(p), 4) for c, p in zip(model.pipeline.classes_, proba, strict=True)}
        return self._finish(model, started, species, probabilities, features.model_dump())

    # ---------------------------------------------------------------- spam

    def predict_spam(self, text: str) -> Prediction:
        model = self.get("spam")
        started = time.perf_counter()
        proba = model.pipeline.predict_proba([text])[0]
        spam_prob = float(proba[1])
        label = "spam" if spam_prob >= 0.5 else "ham"
        details = {"top_terms": self.explain_spam(model, text)}
        logged = {"length": len(text), "n_words": len(text.split())}  # never log the text itself
        return self._finish(
            model,
            started,
            label,
            {"ham": round(1 - spam_prob, 4), "spam": round(spam_prob, 4)},
            logged,
            details,
        )

    @staticmethod
    def explain_spam(model: LoadedModel, text: str, k: int = 5) -> list[dict[str, Any]]:
        """Words in ``text`` that pushed the score towards spam or ham (most influential first)."""
        tfidf = model.pipeline.named_steps["tfidf"]
        clf = model.pipeline.named_steps["clf"]
        vec = tfidf.transform([text])
        if vec.nnz == 0:
            return []
        if hasattr(clf, "coef_"):  # logistic regression: weight * tf-idf value
            weights = clf.coef_[0]
        else:  # naive Bayes: log-odds of the word under spam vs ham
            weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        names = tfidf.get_feature_names_out()
        idx = vec.indices
        contrib = np.asarray(vec.data) * weights[idx]
        order = np.argsort(-np.abs(contrib))[:k]
        return [
            {
                "term": str(names[idx[i]]),
                "push": "spam" if contrib[i] > 0 else "ham",
                "weight": round(float(contrib[i]), 4),
            }
            for i in order
        ]

    # --------------------------------------------------------------- house

    def predict_house(self, features: HouseFeatures | dict[str, float]) -> Prediction:
        model = self.get("house")
        started = time.perf_counter()
        values = features.model_dump() if isinstance(features, HouseFeatures) else dict(features)
        row = pd.DataFrame([values])[HOUSE_FEATURES]
        value = float(model.pipeline.predict(row)[0])
        # Tree models can overshoot the range they were trained on; the data is capped, so clip to it.
        details: dict[str, Any] = {
            "unit": "USD",
            "note": "California block-group median house value (1990 census data)",
        }
        target_range = model.metadata.get("target_range")
        if target_range:
            clipped = float(np.clip(value, target_range[0], target_range[1]))
            if clipped != value:
                details["clipped_to_training_range"] = True
            value = clipped
        usd = round(value * 100_000, 2)
        return self._finish(model, started, usd, None, values, details)

    def predict_house_partial(self, partial: dict[str, float | None]) -> tuple[Prediction, list[str]]:
        """Used by the chat agent: fill features the user did not give with training medians."""
        defaults = self.get("house").metadata["feature_defaults"]
        given = {k: v for k, v in partial.items() if k in HOUSE_FEATURES and v is not None}
        assumed = [k for k in HOUSE_FEATURES if k not in given]
        merged = HouseFeatures(**{**defaults, **given})
        return self.predict_house(merged), assumed

    # ---------------------------------------------------------- monitoring

    def drift(self, name: str) -> dict[str, Any]:
        self.get(name)  # raises if untrained
        return self.logger.drift_report(name, self.settings.models_dir / name / "reference.csv")
