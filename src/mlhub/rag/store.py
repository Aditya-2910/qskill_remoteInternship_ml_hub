"""A small retrieval-augmented-generation knowledge base.

* Documents are markdown files (``knowledge/*.md``) plus "model cards" generated from the trained
  models' metadata, so the assistant can also answer "how accurate is the spam model?".
* Each ``## section`` becomes a chunk (long sections are split by paragraph).
* Chunks are embedded with an Ollama embedding model and searched by cosine similarity.
  If embeddings are not available the store falls back to TF-IDF, so retrieval always works.

The vector store is a plain numpy matrix. For a few dozen chunks that is faster and simpler than
running a database, and it is easy to swap for Chroma or FAISS once the corpus grows.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from mlhub.schemas import SourceChunk

log = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]
MAX_CHUNK_CHARS = 900
# Cosine scores are not comparable across backends, hence one default threshold for each.
MIN_SCORE = {"embeddings": 0.35, "tfidf": 0.08}


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


def chunk_markdown(name: str, text: str) -> list[Chunk]:
    """Split a markdown document into chunks that start with their heading."""
    title = ""
    chunks: list[Chunk] = []
    for block in re.split(r"(?m)^(?=#{1,2} )", text):
        block = block.strip()
        if not block:
            continue
        heading, _, body = block.partition("\n")
        heading = heading.lstrip("# ").strip()
        if block.startswith("# ") and not block.startswith("## "):
            title = heading
            if not body.strip():
                continue
        body = body.strip()
        prefix = f"{title}: {heading}" if title and heading != title else heading
        current = ""
        for para in re.split(r"\n\s*\n", body):
            if current and len(current) + len(para) > MAX_CHUNK_CHARS:
                chunks.append(Chunk(f"{name} / {heading}", f"{prefix}\n{current.strip()}"))
                current = ""
            current += para + "\n\n"
        if current.strip():
            chunks.append(Chunk(f"{name} / {heading}", f"{prefix}\n{current.strip()}"))
    return chunks


def model_card_docs(all_metadata: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Turn trained-model metadata into short text documents the assistant can retrieve."""
    docs: dict[str, str] = {}
    for name, meta in all_metadata.items():
        metrics = ", ".join(f"{k} {v:.3f}" for k, v in meta.get("test_metrics", {}).items())
        cv = ", ".join(f"{k} {v:.3f}" for k, v in meta.get("cv_scores", {}).items())
        docs[f"model card {name}"] = (
            f"## {name} model card\n"
            f"The {name} model is a {meta.get('algorithm')} pipeline for {meta.get('task')}, "
            f"version {meta.get('version')}, trained on {meta.get('n_train')} examples and evaluated "
            f"on {meta.get('n_test')} held-out test examples. Data source: {meta.get('data_source')}. "
            f"Test metrics: {metrics}. Cross-validation scores of the candidate models "
            f"({meta.get('cv_metric')}): {cv}."
        )
    return docs


class KnowledgeBase:
    def __init__(
        self,
        knowledge_dir: Path,
        embed_fn: EmbedFn | None = None,
        embed_model: str = "",
        cache_dir: Path | None = None,
        extra_docs: dict[str, str] | None = None,
    ):
        self.embed_fn = embed_fn
        self.embed_model = embed_model
        # nomic-embed-text was trained with task prefixes and retrieves noticeably better with them
        nomic = "nomic" in embed_model.lower()
        self._doc_prefix = "search_document: " if nomic else ""
        self._query_prefix = "search_query: " if nomic else ""
        self.cache_dir = cache_dir
        self.chunks: list[Chunk] = []
        for path in sorted(Path(knowledge_dir).glob("*.md")):
            self.chunks += chunk_markdown(path.stem, path.read_text(encoding="utf-8"))
        for name, text in (extra_docs or {}).items():
            self.chunks += chunk_markdown(name, text)
        if not self.chunks:
            raise ValueError(f"No knowledge documents found in {knowledge_dir}")

        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._tfidf_matrix = self._tfidf.fit_transform([c.text for c in self.chunks])
        self._embeddings: np.ndarray | None = None
        self.backend = "tfidf"
        if embed_fn is not None:
            self._try_build_embeddings()

    # ------------------------------------------------------------------ building

    def _cache_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(
            (self.embed_model + "\x00" + "\x00".join(c.text for c in self.chunks)).encode()
        ).hexdigest()[:16]
        return Path(self.cache_dir) / f"embeddings-{digest}.npy"

    def _try_build_embeddings(self) -> None:
        cache = self._cache_path()
        try:
            if cache and cache.exists():
                matrix = np.load(cache)
            else:
                vectors: list[list[float]] = []
                texts = [self._doc_prefix + c.text for c in self.chunks]
                for i in range(0, len(texts), 16):  # small batches keep requests light
                    vectors += self.embed_fn(texts[i : i + 16])
                matrix = np.asarray(vectors, dtype=np.float32)
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, matrix)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._embeddings = matrix / np.clip(norms, 1e-12, None)
            self.backend = "embeddings"
        except Exception as exc:  # Ollama down / embedding model missing: TF-IDF still works
            log.warning("Embeddings unavailable (%s); using TF-IDF retrieval.", exc)
            self._embeddings = None
            self.backend = "tfidf"

    # ------------------------------------------------------------------- search

    def _scores(self, query: str) -> tuple[np.ndarray, str]:
        if self._embeddings is not None and self.embed_fn is not None:
            try:
                q = np.asarray(self.embed_fn([self._query_prefix + query])[0], dtype=np.float32)
                q /= max(float(np.linalg.norm(q)), 1e-12)
                return self._embeddings @ q, "embeddings"
            except Exception as exc:
                log.warning("Query embedding failed (%s); using TF-IDF for this query.", exc)
        q_vec = self._tfidf.transform([query])
        return linear_kernel(q_vec, self._tfidf_matrix).ravel(), "tfidf"

    def search(self, query: str, k: int = 3, min_score: float | None = None) -> list[SourceChunk]:
        scores, backend = self._scores(query)
        threshold = MIN_SCORE[backend] if min_score is None else min_score
        results = []
        for i in np.argsort(-scores)[:k]:
            if scores[i] >= threshold:
                chunk = self.chunks[i]
                results.append(SourceChunk(source=chunk.source, text=chunk.text, score=round(float(scores[i]), 4)))
        return results
