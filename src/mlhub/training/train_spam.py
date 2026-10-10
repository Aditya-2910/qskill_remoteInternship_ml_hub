"""Task 2: spam mail (SMS) detector."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from mlhub.config import Settings, get_settings
from mlhub.datasets import load_spam_df
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
NAME = "spam"
LABELS = ["ham", "spam"]  # index == class id, "spam" is the positive class


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data: drop empty rows and exact duplicates (they would leak across the split)."""
    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].drop_duplicates(subset="text")
    df["y"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)


def split_spam(df: pd.DataFrame, random_state: int = 42):
    """Deterministic stratified split. The benchmark script re-uses this to get the same test set."""
    df = prepare(df)
    return train_test_split(df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=random_state)


def make_vectorizer() -> TfidfVectorizer:
    # Preprocessing from the task sheet, done inside the pipeline so it is fit on training data only:
    # lowercasing, tokenization, stop-word removal, then TF-IDF weighting (uni- and bi-grams).
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def explore(df: pd.DataFrame, out_dir) -> list:
    df = df.assign(length=df["text"].str.len())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["label"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#4c72b0", "#dd8452"][: len(counts)])
    axes[0].set_title("Class balance")
    for label, group in df.groupby("label"):
        axes[1].hist(group["length"].clip(upper=400), bins=30, alpha=0.6, label=label)
    axes[1].set_title("Message length (characters)")
    axes[1].legend()
    return [save_figure(fig, out_dir / "spam_eda.png")]


def train(settings: Settings | None = None) -> TrainingResult:
    settings = settings or get_settings()
    settings.ensure_dirs()
    out_dir = ensure_reports(settings, NAME)

    raw, source = load_spam_df(settings)
    df = prepare(raw)
    log.info("Spam: %d messages after cleaning, %.1f%% spam (%s)", len(df), 100 * df["y"].mean(), source)
    plots = explore(df, out_dir)

    X_train, X_test, y_train, y_test = split_spam(raw, settings.random_state)

    candidates = {
        "naive_bayes": Pipeline([("tfidf", make_vectorizer()), ("clf", MultinomialNB(alpha=0.1))]),
        "logistic_regression": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("clf", LogisticRegression(C=10, max_iter=1000, class_weight="balanced")),
            ]
        ),
    }

    tracker = Tracker(settings, "mlhub-spam")
    with tracker.run("spam-training") as parent:
        tracker.params(
            {"dataset": source, "n_train": len(X_train), "n_test": len(X_test), "random_state": settings.random_state}
        )
        # F1 on the spam class: accuracy alone is misleading with ~13% spam.
        best_name, cv_scores = select_best(candidates, X_train, y_train, "f1", tracker)

        pipeline = candidates[best_name].fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", pos_label=1, zero_division=0
        )
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        plots.append(plot_confusion(cm, LABELS, f"Spam: {best_name}", out_dir / "spam_confusion.png"))
        (out_dir / "classification_report.txt").write_text(
            classification_report(y_test, pred, target_names=LABELS, zero_division=0)
        )

        tracker.params({"best_model": best_name})
        tracker.metrics({f"test_{k}": v for k, v in metrics.items()})
        for plot in plots:
            tracker.artifact(plot, "plots")
        tracker.model(pipeline, "model", "mlhub-spam")

        metadata = {
            "task": "classification",
            "algorithm": best_name,
            "data_source": source,
            "feature_names": ["text"],
            "classes": LABELS,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "cv_scores": cv_scores,
            "cv_metric": "f1",
            "test_metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "mlflow_run_id": parent.info.run_id if parent else None,
        }

    sample = X_train.sample(n=min(len(X_train), 1000), random_state=settings.random_state)
    reference = pd.DataFrame({"length": sample.str.len(), "n_words": sample.str.split().str.len()})
    model_dir = save_model_dir(settings, NAME, pipeline, metadata, reference, plots)
    log.info("Spam: best=%s test F1=%.3f -> %s", best_name, metrics["f1"], model_dir)
    return TrainingResult(NAME, "classification", best_name, cv_scores, metrics, metadata, model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()
