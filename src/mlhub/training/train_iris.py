"""Task 1: Iris flower classification."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlhub.config import Settings, get_settings
from mlhub.datasets import IRIS_FEATURES, IRIS_SPECIES, load_iris_df
from mlhub.training.common import (
    Tracker,
    TrainingResult,
    ensure_reports,
    plot_confusion,
    save_figure,
    save_model_dir,
    select_best,
)

log = logging.getLogger(__name__)
NAME = "iris"


def explore(df: pd.DataFrame, out_dir) -> list:
    """Step 1 of the task: look at the data before modelling."""
    colors = {"setosa": "#1f77b4", "versicolor": "#ff7f0e", "virginica": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, feature in zip(axes.ravel(), IRIS_FEATURES, strict=True):
        for species, group in df.groupby("species"):
            ax.hist(group[feature], bins=15, alpha=0.6, label=species, color=colors[species])
        ax.set_title(feature.replace("_", " "))
        ax.set_xlabel("cm")
    axes[0, 0].legend()
    fig.suptitle("Iris feature distributions by species")
    hist = save_figure(fig, out_dir / "iris_histograms.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for species, group in df.groupby("species"):
        ax.scatter(
            group["petal_length"],
            group["petal_width"],
            label=species,
            color=colors[species],
            alpha=0.8,
            edgecolor="white",
        )
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")
    ax.set_title("Petal length vs width")
    ax.legend()
    scatter = save_figure(fig, out_dir / "iris_scatter.png")
    return [hist, scatter]


def train(settings: Settings | None = None) -> TrainingResult:
    settings = settings or get_settings()
    settings.ensure_dirs()
    out_dir = ensure_reports(settings, NAME)

    df, source = load_iris_df()
    log.info("Iris: %d rows, %d features (%s)", len(df), len(IRIS_FEATURES), source)
    plots = explore(df, out_dir)

    X, y = df[IRIS_FEATURES], df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=settings.random_state
    )

    candidates = {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]),
        "knn": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "decision_tree": Pipeline([("clf", DecisionTreeClassifier(max_depth=4, random_state=settings.random_state))]),
    }

    tracker = Tracker(settings, "mlhub-iris")
    with tracker.run("iris-training") as parent:
        tracker.params(
            {
                "dataset": source,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_size": 0.2,
                "random_state": settings.random_state,
            }
        )
        best_name, cv_scores = select_best(candidates, X_train, y_train, "accuracy", tracker)

        pipeline = candidates[best_name].fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }
        cm = confusion_matrix(y_test, pred, labels=IRIS_SPECIES)
        plots.append(plot_confusion(cm, IRIS_SPECIES, f"Iris: {best_name}", out_dir / "iris_confusion.png"))
        (out_dir / "classification_report.txt").write_text(classification_report(y_test, pred, zero_division=0))

        tracker.params({"best_model": best_name})
        tracker.metrics({f"test_{k}": v for k, v in metrics.items()})
        for plot in plots:
            tracker.artifact(plot, "plots")
        tracker.model(pipeline, "model", "mlhub-iris")

        metadata = {
            "task": "classification",
            "algorithm": best_name,
            "data_source": source,
            "feature_names": IRIS_FEATURES,
            "classes": IRIS_SPECIES,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "cv_scores": cv_scores,
            "cv_metric": "accuracy",
            "test_metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "mlflow_run_id": parent.info.run_id if parent else None,
        }

    model_dir = save_model_dir(
        settings,
        NAME,
        pipeline,
        metadata,
        X_train.sample(n=min(len(X_train), 500), random_state=settings.random_state),
        plots,
    )
    log.info("Iris: best=%s test accuracy=%.3f -> %s", best_name, metrics["accuracy"], model_dir)
    return TrainingResult(NAME, "classification", best_name, cv_scores, metrics, metadata, model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()
