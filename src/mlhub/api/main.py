"""FastAPI backend: model endpoints, chat agent, monitoring.

Run:  uvicorn mlhub.api.main:create_app --factory --reload
Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from mlhub import __version__
from mlhub.config import Settings, get_settings
from mlhub.llm.agent import Agent
from mlhub.llm.ollama_client import LLMUnavailable, OllamaClient
from mlhub.rag.store import KnowledgeBase, model_card_docs
from mlhub.schemas import (
    ChatRequest,
    ChatResponse,
    HouseFeatures,
    IrisFeatures,
    Prediction,
    SourceChunk,
    SpamRequest,
)
from mlhub.services import MODEL_NAMES, ModelNotAvailable, ModelService

log = logging.getLogger("mlhub.api")


def create_app(settings: Settings | None = None, llm: Any | None = None) -> FastAPI:
    """Application factory. Tests pass their own ``settings`` and a fake ``llm``."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    models = ModelService(settings)
    llm = llm or OllamaClient(
        settings.ollama_host, settings.ollama_model, settings.ollama_embed_model, settings.ollama_timeout
    )

    # The knowledge base is built on first use (after models exist and Ollama is up) and cached.
    kb_lock = threading.Lock()
    cache: dict[str, Any] = {"agent": None, "kb": None}

    def get_kb() -> KnowledgeBase:
        with kb_lock:
            if cache["kb"] is None:
                cache["kb"] = KnowledgeBase(
                    settings.knowledge_dir,
                    embed_fn=getattr(llm, "embed", None),
                    embed_model=settings.ollama_embed_model,
                    cache_dir=settings.cache_dir,
                    extra_docs=model_card_docs(models.all_metadata()),
                )
            return cache["kb"]

    def get_agent() -> Agent:
        kb = get_kb()
        with kb_lock:
            if cache["agent"] is None:
                cache["agent"] = Agent(llm, models, kb)
            return cache["agent"]

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")

    app = FastAPI(
        title="ML Hub API",
        version=__version__,
        description="Iris, spam and house-price models plus a local-LLM assistant (tools + RAG).",
    )
    app.state.settings = settings
    app.state.models = models
    app.state.llm = llm

    @app.exception_handler(ModelNotAvailable)
    async def _model_missing(_request, exc: ModelNotAvailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMUnavailable)
    async def _llm_missing(_request, exc: LLMUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # --------------------------------------------------------------------- public

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, Any]:
        available = models.available()
        ollama: dict[str, Any] = {"host": settings.ollama_host, "model": settings.ollama_model}
        try:
            reachable = llm.is_available()
            ollama["reachable"] = reachable
            ollama["model_available"] = bool(reachable and llm.has_model())
        except Exception:  # never let a health probe crash
            ollama["reachable"] = False
            ollama["model_available"] = False
        return {
            "status": "ok" if all(available.values()) else "degraded",
            "version": __version__,
            "models": available,
            "ollama": ollama,
        }

    # ------------------------------------------------------------------ protected

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

    @v1.get("/models", tags=["models"])
    def list_models() -> dict[str, Any]:
        return models.all_metadata()

    @v1.get("/models/{name}/plots/{filename}", tags=["models"])
    def get_plot(name: str, filename: str) -> FileResponse:
        if name not in MODEL_NAMES or not re.fullmatch(r"[\w\-]+\.png", filename):
            raise HTTPException(status_code=404, detail="Unknown plot")
        path = settings.models_dir / name / "plots" / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Unknown plot")
        return FileResponse(path, media_type="image/png")

    @v1.post("/predict/iris", response_model=Prediction, tags=["predict"])
    def predict_iris(features: IrisFeatures) -> Prediction:
        return models.predict_iris(features)

    @v1.post("/predict/spam", response_model=Prediction, tags=["predict"])
    def predict_spam(request: SpamRequest) -> Prediction:
        return models.predict_spam(request.text)

    @v1.post("/predict/house", response_model=Prediction, tags=["predict"])
    def predict_house(features: HouseFeatures) -> Prediction:
        return models.predict_house(features)

    @v1.post("/chat", response_model=ChatResponse, tags=["assistant"])
    def chat(request: ChatRequest) -> ChatResponse:
        return get_agent().run(request.message, request.history)

    @v1.get("/knowledge/search", response_model=list[SourceChunk], tags=["assistant"])
    def knowledge_search(q: str = Query(..., min_length=2), k: int = Query(3, ge=1, le=10)) -> list[SourceChunk]:
        return get_kb().search(q, k=k)

    @v1.get("/monitoring/summary", tags=["monitoring"])
    def monitoring_summary() -> dict[str, Any]:
        return models.logger.summary()

    @v1.get("/monitoring/drift/{name}", tags=["monitoring"])
    def monitoring_drift(name: str) -> dict[str, Any]:
        if name not in MODEL_NAMES:
            raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
        return models.drift(name)

    @v1.post("/admin/reload", tags=["admin"])
    def reload() -> dict[str, str]:
        """Pick up freshly trained models and rebuild the knowledge base."""
        models.reload()
        with kb_lock:
            cache["kb"] = None
            cache["agent"] = None
        return {"status": "reloaded"}

    app.include_router(v1)
    return app
