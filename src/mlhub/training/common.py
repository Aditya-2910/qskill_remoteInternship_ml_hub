"""Shared training helpers: candidate selection, MLflow tracking and artifact saving."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import BaseEstimator, clone  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

from mlhub.config import Settings  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """What a training script returns (and what ends up in ``metadata.json``)."""

    name: str
    task: str
    best_model: str
    cv_scores: dict[str, float]
    test_metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    model_dir: Path | None = None


# --------------------------------------------------------------------------- MLflow


class Tracker:
    """Thin wrapper so training code reads the same with or without MLflow."""

    def __init__(self, settings: Settings, experiment: str):
        self.enabled = settings.mlflow_enabled
        self._mlflow = None
        if self.enabled:
            import mlflow

            settings.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=settings.mlflow_artifact_root.resolve().as_uri())
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow

    @contextlib.contextmanager
    def run(self, name: str, nested: bool = False) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=name, nested=nested) as run:
            yield run

    def params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def metrics(self, metrics: dict[str, float]) -> None:
        if self.enabled:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def tags(self, tags: dict[str, str]) -> None:
        if self.enabled:
            self._mlflow.set_tags(tags)

    def artifact(self, path: Path, folder: str | None = None) -> None:
        if self.enabled and path.exists():
            self._mlflow.log_artifact(str(path), artifact_path=folder)

    def model(self, pipeline: BaseEstimator, name: str, registered_name: str) -> None:
        if self.enabled:
            # cloudpickle: MLflow's newer default (skops) refuses some sklearn types such as KNN trees.
            self._mlflow.sklearn.log_model(
                pipeline,
                name=name,
                registered_model_name=registered_name,
                serialization_format="cloudpickle",
            )


# --------------------------------------------------------------------- model selection


def select_best(
    candidates: dict[str, BaseEstimator],
    X_train: Any,
    y_train: Any,
    scoring: str,
    tracker: Tracker,
    cv: int = 5,
) -> tuple[str, dict[str, float]]:
    """Cross-validate every candidate on the *training* set only and return the winner.

    The test set is never touched here, so choosing the model cannot leak information from it.
    Each candidate is logged as a nested MLflow run.
    """
    scores: dict[str, float] = {}
    for name, estimator in candidates.items():
        with tracker.run(f"cv-{name}", nested=True):
            folds = cross_val_score(clone(estimator), X_train, y_train, cv=cv, scoring=scoring)
            scores[name] = float(np.mean(folds))
            tracker.params({"candidate": name, "scoring": scoring, "cv_folds": cv})
            tracker.metrics({f"cv_{scoring}_mean": scores[name], f"cv_{scoring}_std": float(np.std(folds))})
        log.info("  CV %-22s %s = %.4f", name, scoring, scores[name])
    best = max(scores, key=scores.get)  # all scorers are "higher is better" in sklearn
    return best, scores


# --------------------------------------------------------------------------- plotting


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    return save_figure(fig, path)


# ----------------------------------------------------------------------- persistence


def save_model_dir(
    settings: Settings,
    name: str,
    pipeline: BaseEstimator,
    metadata: dict[str, Any],
    reference: pd.DataFrame | None,
    plots: list[Path],
) -> Path:
    """Write ``models/<name>/{model.joblib, metadata.json, reference.csv, plots/}``."""
    out = settings.models_dir / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "model.joblib", compress=3)
    metadata = {
        **metadata,
        "name": name,
        "version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if reference is not None:
        reference.to_csv(out / "reference.csv", index=False)
    for plot in plots:
        target = out / "plots" / plot.name
        if plot.resolve() != target.resolve():
            target.write_bytes(plot.read_bytes())
    return out


def ensure_reports(settings: Settings, name: str) -> Path:
    path = settings.reports_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


TrainFn = Callable[..., TrainingResult]
