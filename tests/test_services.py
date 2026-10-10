from __future__ import annotations

import json

import pandas as pd
import pytest
from conftest import make_settings

from mlhub.schemas import HouseFeatures, IrisFeatures
from mlhub.services import ModelNotAvailable, ModelService

SETOSA = IrisFeatures(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
VIRGINICA = IrisFeatures(sepal_length=6.7, sepal_width=3.0, petal_length=5.8, petal_width=2.2)


def test_iris_predictions(fresh_settings):
    svc = ModelService(fresh_settings)
    setosa = svc.predict_iris(SETOSA)
    assert setosa.prediction == "setosa"
    assert setosa.probabilities["setosa"] > 0.8
    assert sum(setosa.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert svc.predict_iris(VIRGINICA).prediction == "virginica"


def test_spam_prediction_and_explanation(fresh_settings):
    svc = ModelService(fresh_settings)
    spam = svc.predict_spam("WINNER!! Claim your free prize now, call 07123456789")
    assert spam.prediction == "spam"
    assert spam.probabilities["spam"] > 0.5
    terms = spam.details["top_terms"]
    assert terms and all({"term", "push", "weight"} <= set(t) for t in terms)
    assert any(t["push"] == "spam" for t in terms)

    ham = svc.predict_spam("Are we still meeting for lunch tomorrow?")
    assert ham.prediction == "ham"


def test_spam_of_only_stopwords_does_not_crash(fresh_settings):
    pred = ModelService(fresh_settings).predict_spam("the and of")
    assert pred.details["top_terms"] == []


def test_house_prediction_and_partial(fresh_settings):
    svc = ModelService(fresh_settings)
    full = HouseFeatures(**svc.metadata("house")["feature_defaults"])
    pred = svc.predict_house(full)
    assert 10_000 < pred.prediction < 600_000

    rich, assumed = svc.predict_house_partial({"MedInc": 12.0})
    poor, _ = svc.predict_house_partial({"MedInc": 1.5})
    assert rich.prediction > poor.prediction
    assert "MedInc" not in assumed and len(assumed) == 7


def test_untrained_model_raises(tmp_path):
    svc = ModelService(make_settings(tmp_path))
    assert svc.available() == {"iris": False, "spam": False, "house": False}
    with pytest.raises(ModelNotAvailable, match="make train"):
        svc.predict_iris(SETOSA)


def test_spam_text_is_never_logged(fresh_settings):
    svc = ModelService(fresh_settings)
    secret = "my secret bank pin is 1234 claim prize"
    svc.predict_spam(secret)
    raw = (fresh_settings.logs_dir / "predictions.jsonl").read_text()
    assert "secret" not in raw and "1234" not in raw
    rec = json.loads(raw.splitlines()[-1])
    assert rec["inputs"] == {"length": len(secret), "n_words": len(secret.split())}


def test_monitoring_summary(fresh_settings):
    svc = ModelService(fresh_settings)
    for _ in range(3):
        svc.predict_iris(SETOSA)
    svc.predict_iris(VIRGINICA)
    summary = svc.logger.summary()
    assert summary["total_predictions"] == 4
    assert summary["models"]["iris"]["count"] == 4
    assert summary["models"]["iris"]["prediction_counts"] == {"setosa": 3, "virginica": 1}
    assert summary["models"]["iris"]["latency_ms_p95"] >= 0


def test_drift_detection(fresh_settings):
    svc = ModelService(fresh_settings)
    assert svc.drift("iris")["status"] == "not_enough_data"

    ref = pd.read_csv(fresh_settings.models_dir / "iris" / "reference.csv")
    # 1) live traffic that looks like the training data -> no drift
    for row in ref.sample(60, random_state=1).to_dict("records"):
        svc.logger.log("iris", row, "setosa", 1.0)
    calm = svc.drift("iris")
    assert calm["status"] == "ok" and calm["drift_detected"] is False

    # 2) all petals suddenly huge -> drift on the petal features
    svc2 = ModelService(make_settings(fresh_settings.base_dir / "other"))
    svc2.settings.models_dir.parent.mkdir(exist_ok=True)
    import shutil

    shutil.copytree(fresh_settings.models_dir, svc2.settings.models_dir)
    for _ in range(60):
        svc2.logger.log(
            "iris", {"sepal_length": 5.8, "sepal_width": 3.0, "petal_length": 9.5, "petal_width": 4.0}, "virginica", 1.0
        )
    shifted = svc2.drift("iris")
    drifted = {f["feature"] for f in shifted["features"] if f["drift"]}
    assert shifted["drift_detected"] and {"petal_length", "petal_width"} <= drifted


def test_house_prediction_is_clipped_to_training_range(fresh_settings):
    svc = ModelService(fresh_settings)
    low, high = svc.metadata("house")["target_range"]
    defaults = dict(svc.metadata("house")["feature_defaults"])
    extreme = svc.predict_house(HouseFeatures(**{**defaults, "MedInc": 15.0, "AveRooms": 12.0, "Longitude": -122.0}))
    assert low * 100_000 <= extreme.prediction <= high * 100_000
    cheap = svc.predict_house(HouseFeatures(**{**defaults, "MedInc": 0.5, "Longitude": -115.0}))
    assert cheap.prediction >= low * 100_000
