"""Central configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """All runtime settings in one place.

    Everything lives under ``base_dir`` so that a Docker volume (or a temp dir in tests)
    can hold models, logs, cached data and MLflow state.
    """

    base_dir: Path = field(default_factory=lambda: Path(os.getenv("MLHUB_HOME", Path.cwd())))

    # Local LLM (Ollama)
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    ollama_embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    ollama_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "120")))

    # API
    api_key: str | None = field(default_factory=lambda: os.getenv("MLHUB_API_KEY") or None)
    api_url: str = field(default_factory=lambda: os.getenv("MLHUB_API_URL", "http://localhost:8000"))

    # Training
    offline: bool = field(default_factory=lambda: _env_bool("MLHUB_OFFLINE", False))
    mlflow_enabled: bool = field(default_factory=lambda: _env_bool("MLHUB_MLFLOW", True))
    random_state: int = 42

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.base_dir / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / ".cache"

    @property
    def mlflow_tracking_uri(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{self.base_dir / 'mlflow.db'}")

    @property
    def mlflow_artifact_root(self) -> Path:
        return self.base_dir / "mlartifacts"

    @property
    def knowledge_dir(self) -> Path:
        return Path(__file__).parent / "rag" / "knowledge"

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.data_dir, self.logs_dir, self.reports_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
