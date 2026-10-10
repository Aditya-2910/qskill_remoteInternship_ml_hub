from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
from conftest import make_settings

from mlhub.datasets import HOUSE_FEATURES, load_house_df, load_spam_df, make_sample_sms, parse_sms_zip
from mlhub.training import train_iris
from mlhub.training.train_spam import prepare, split_spam


def _meta(settings, name):
    return json.loads((settings.models_dir / name / "metadata.json").read_text())


def test_all_artifacts_written(trained_settings):
    for name in ("iris", "spam", "house"):
        d = trained_settings.models_dir / name
        assert (d / "model.joblib").exists()
        assert (d / "metadata.json").exists()
        assert (d / "reference.csv").exists()
        assert list((d / "plots").glob("*.png")), f"no plots saved for {name}"


def test_iris_quality(trained_settings):
    meta = _meta(trained_settings, "iris")
    assert meta["test_metrics"]["accuracy"] >= 0.85
    assert set(meta["cv_scores"]) == {"logistic_regression", "knn", "decision_tree"}
    assert meta["algorithm"] == max(meta["cv_scores"], key=meta["cv_scores"].get)


def test_spam_quality_and_metadata(trained_settings):
    meta = _meta(trained_settings, "spam")
    assert meta["test_metrics"]["f1"] >= 0.9
    assert "synthetic" in meta["data_source"]  # offline runs must be labelled as such
    assert meta["classes"] == ["ham", "spam"]


def test_house_quality_and_defaults(trained_settings):
    meta = _meta(trained_settings, "house")
    assert meta["test_metrics"]["r2"] > 0.5
    assert "baseline_linear_mse" in meta["test_metrics"]
    assert set(meta["feature_defaults"]) == set(HOUSE_FEATURES)


def test_spam_split_has_no_overlap():
    df = make_sample_sms()
    X_train, X_test, y_train, y_test = split_spam(df)
    assert not set(X_train) & set(X_test), "the same message must not be in both train and test"
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    # stratified: class ratio preserved
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_prepare_drops_duplicates_and_blanks():
    df = pd.DataFrame({"label": ["ham", "ham", "spam", "spam"], "text": ["hi", "hi", "win now", "  "]})
    out = prepare(df)
    assert list(out["text"]) == ["hi", "win now"]
    assert list(out["y"]) == [0, 1]


def test_parse_uci_zip():
    raw = "ham\tHello there\nspam\tWIN a prize now\nbroken line without tab\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SMSSpamCollection", raw)
        zf.writestr("readme", "ignored")
    df = parse_sms_zip(buf.getvalue())
    assert df.to_dict("records") == [
        {"label": "ham", "text": "Hello there"},
        {"label": "spam", "text": "WIN a prize now"},
    ]


def test_offline_loaders_fall_back(tmp_path):
    settings = make_settings(tmp_path, offline=True)
    spam, source = load_spam_df(settings)
    assert "synthetic" in source and {"label", "text"} <= set(spam.columns)
    house, source = load_house_df(settings)
    assert "synthetic" in source and set(HOUSE_FEATURES) <= set(house.columns)


def test_download_failure_falls_back(tmp_path, monkeypatch):
    import mlhub.datasets as ds

    def boom(cache):
        raise OSError("no network")

    monkeypatch.setattr(ds, "_download_sms", boom)
    monkeypatch.setattr(ds, "fetch_california_housing", boom)
    settings = make_settings(tmp_path, offline=False)
    assert "synthetic" in load_spam_df(settings)[1]
    assert "synthetic" in load_house_df(settings)[1]


def test_mlflow_tracking_and_registry(tmp_path):
    """With MLflow on, a training run creates an experiment, nested CV runs and a registered model."""
    import mlflow

    settings = make_settings(tmp_path, mlflow_enabled=True)
    result = train_iris.train(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name("mlhub-iris")
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) == 1 + 3  # parent + one nested run per candidate
    parent = runs[runs["run_id"] == result.metadata["mlflow_run_id"]].iloc[0]
    assert parent["metrics.test_accuracy"] == result.test_metrics["accuracy"]
    assert [m.name for m in mlflow.MlflowClient().search_registered_models()] == ["mlhub-iris"]
