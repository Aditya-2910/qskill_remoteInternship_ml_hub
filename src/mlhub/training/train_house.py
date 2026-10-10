"""Task 3: house price prediction (regression)."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlhub.config import Settings, get_settings
from mlhub.datasets import HOUSE_FEATURES, HOUSE_TARGET, load_house_df
from mlhub.training.common import (
    Tracker,
    TrainingResult,
    ensure_reports,
    save_figure,
    save_model_dir,
    select_best,
)

log = logging.getLogger(__name__)
NAME = "house"


def _linear_pipeline(estimator) -> Pipeline:
    # Missing values -> median, then normalisation (both fit on training data only).
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("reg", estimator),
        ]
    )


def explore(df: pd.DataFrame, out_dir) -> list:
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for ax, col in zip(axes.ravel(), HOUSE_FEATURES + [HOUSE_TARGET], strict=True):
        ax.hist(df[col].dropna(), bins=40, color="#4c72b0")
        ax.set_title(col)
    fig.suptitle("Feature and target distributions")
    fig.tight_layout()
    hist = save_figure(fig, out_dir / "house_histograms.png")

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    heat = save_figure(fig, out_dir / "house_correlation.png")
    return [hist, heat]


def train(settings: Settings | None = None) -> TrainingResult:
    settings = settings or get_settings()
    settings.ensure_dirs()
    out_dir = ensure_reports(settings, NAME)

    df, source = load_house_df(settings)
    log.info("House: %d rows, %d missing cells (%s)", len(df), int(df.isna().sum().sum()), source)
    plots = explore(df, out_dir)

    X, y = df[HOUSE_FEATURES], df[HOUSE_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=settings.random_state)

    candidates = {
        "linear_regression": _linear_pipeline(LinearRegression()),
        "ridge": _linear_pipeline(Ridge(alpha=1.0)),
        "hist_gradient_boosting": Pipeline(
            [("reg", HistGradientBoostingRegressor(max_iter=200, random_state=settings.random_state))]
        ),
    }

    tracker = Tracker(settings, "mlhub-house")
    with tracker.run("house-training") as parent:
        tracker.params(
            {"dataset": source, "n_train": len(X_train), "n_test": len(X_test), "random_state": settings.random_state}
        )
        best_name, cv_scores = select_best(candidates, X_train, y_train, "neg_root_mean_squared_error", tracker)

        # The task sheet asks for plain Linear Regression + MSE; always report that baseline too.
        baseline = candidates["linear_regression"].fit(X_train, y_train)
        baseline_mse = float(mean_squared_error(y_test, baseline.predict(X_test)))

        pipeline = candidates[best_name].fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        mse = float(mean_squared_error(y_test, pred))
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "baseline_linear_mse": baseline_mse,
        }

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(y_test, pred, s=6, alpha=0.4)
        lim = [float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))]
        ax.plot(lim, lim, color="crimson", lw=1)
        ax.set_xlabel(f"Actual {HOUSE_TARGET} ($100k)")
        ax.set_ylabel("Predicted ($100k)")
        ax.set_title(f"House: {best_name} (R²={metrics['r2']:.2f})")
        plots.append(save_figure(fig, out_dir / "house_pred_vs_actual.png"))

        tracker.params({"best_model": best_name})
        tracker.metrics({f"test_{k}": v for k, v in metrics.items()})
        for plot in plots:
            tracker.artifact(plot, "plots")
        tracker.model(pipeline, "model", "mlhub-house")

        metadata = {
            "task": "regression",
            "algorithm": best_name,
            "data_source": source,
            "feature_names": HOUSE_FEATURES,
            "target": HOUSE_TARGET,
            "target_unit": "USD 100,000",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "cv_scores": cv_scores,
            "cv_metric": "neg_root_mean_squared_error",
            "test_metrics": metrics,
            # used by the chat agent to fill in features the user did not mention
            "target_range": [float(y_train.min()), float(y_train.max())],
            "feature_defaults": {k: float(v) for k, v in X_train.median().items()},
            "mlflow_run_id": parent.info.run_id if parent else None,
        }

    reference = X_train.sample(n=min(len(X_train), 1000), random_state=settings.random_state)
    model_dir = save_model_dir(settings, NAME, pipeline, metadata, reference, plots)
    log.info("House: best=%s test RMSE=%.3f R2=%.3f -> %s", best_name, metrics["rmse"], metrics["r2"], model_dir)
    return TrainingResult(NAME, "regression", best_name, cv_scores, metrics, metadata, model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()
