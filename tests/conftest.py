from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from mlhub.config import Settings
from mlhub.training import train_house, train_iris, train_spam


def make_settings(base: Path, **overrides) -> Settings:
    """Settings that never depend on the developer's environment variables."""
    defaults = dict(base_dir=base, offline=True, mlflow_enabled=False, api_key=None, ollama_host="http://fake-ollama")
    return dataclasses.replace(Settings(), **{**defaults, **overrides})


@pytest.fixture(scope="session")
def trained_settings(tmp_path_factory) -> Settings:
    """Train all three models once (offline, no MLflow) and share the result across tests."""
    settings = make_settings(tmp_path_factory.mktemp("mlhub"))
    train_iris.train(settings)
    train_spam.train(settings)
    train_house.train(settings)
    return settings


@pytest.fixture()
def fresh_settings(trained_settings, tmp_path) -> Settings:
    """A copy of the trained models with an empty log directory, so tests do not see each other's logs."""
    import shutil

    shutil.copytree(trained_settings.models_dir, tmp_path / "models")
    return make_settings(tmp_path)
