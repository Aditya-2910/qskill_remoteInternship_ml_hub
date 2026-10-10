from __future__ import annotations

import numpy as np
import pytest
from fake_ollama import _embed

from mlhub.config import Settings
from mlhub.rag.store import KnowledgeBase, chunk_markdown, model_card_docs

KNOWLEDGE_DIR = Settings().knowledge_dir


def test_chunking_keeps_headings_and_splits_long_sections():
    text = "# Title\n\n## Short\nOne paragraph.\n\n## Long\n" + "\n\n".join(["word " * 100] * 5)
    chunks = chunk_markdown("doc", text)
    assert chunks[0].source == "doc / Short"
    assert chunks[0].text.startswith("Title: Short")
    long_chunks = [c for c in chunks if c.source == "doc / Long"]
    assert len(long_chunks) > 1
    assert all(len(c.text) < 1300 for c in long_chunks)


def test_tfidf_retrieval_finds_the_right_document():
    kb = KnowledgeBase(KNOWLEDGE_DIR)
    assert kb.backend == "tfidf"
    top = kb.search("What are the warning signs of a phishing text message?", k=3)
    assert top and top[0].source.startswith("spam")
    top = kb.search("Which iris species has the largest petals?", k=3)
    assert top[0].source.startswith("iris")
    top = kb.search("What is MedInc, the median income feature?", k=3)
    assert top[0].source.startswith("house_prices")


def test_irrelevant_query_returns_nothing():
    kb = KnowledgeBase(KNOWLEDGE_DIR)
    assert kb.search("qwertyuiop zxcvbnm asdfgh", k=3) == []


def test_embedding_backend_and_cache(tmp_path):
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return [_embed(t) for t in texts]

    kb = KnowledgeBase(KNOWLEDGE_DIR, embed_fn=embed, embed_model="fake", cache_dir=tmp_path)
    assert kb.backend == "embeddings"
    assert kb.search("petal length of setosa versicolor virginica", k=1, min_score=0.0)[0].source.startswith("iris")
    assert len(list(tmp_path.glob("embeddings-*.npy"))) == 1

    calls.clear()
    KnowledgeBase(KNOWLEDGE_DIR, embed_fn=embed, embed_model="fake", cache_dir=tmp_path)
    assert calls == [], "second start must load embeddings from the cache instead of re-embedding"


def test_falls_back_to_tfidf_when_embeddings_fail():
    def broken(_texts):
        raise ConnectionError("ollama is down")

    kb = KnowledgeBase(KNOWLEDGE_DIR, embed_fn=broken)
    assert kb.backend == "tfidf"
    assert kb.search("spam precision recall", k=1)


def test_query_embedding_failure_uses_tfidf_for_that_query():
    state = {"fail": False}

    def flaky(texts):
        if state["fail"]:
            raise ConnectionError("gone")
        return [_embed(t) for t in texts]

    kb = KnowledgeBase(KNOWLEDGE_DIR, embed_fn=flaky)
    assert kb.backend == "embeddings"
    state["fail"] = True
    assert kb.search("largest petals of iris species", k=1)[0].source.startswith("iris")


def test_model_cards_are_searchable():
    meta = {
        "spam": {
            "algorithm": "naive_bayes",
            "task": "classification",
            "version": "1",
            "n_train": 800,
            "n_test": 200,
            "data_source": "UCI",
            "cv_metric": "f1",
            "test_metrics": {"f1": 0.9123},
            "cv_scores": {"naive_bayes": 0.9},
        }
    }
    kb = KnowledgeBase(KNOWLEDGE_DIR, extra_docs=model_card_docs(meta))
    hit = kb.search("how good is the spam model test metrics", k=3)
    assert any("0.912" in h.text for h in hit)


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        KnowledgeBase(tmp_path)


def test_normalised_embeddings_give_cosine_scores():
    kb = KnowledgeBase(KNOWLEDGE_DIR, embed_fn=lambda ts: [_embed(t) for t in ts])
    norms = np.linalg.norm(kb._embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_nomic_models_get_task_prefixes():
    seen = []

    def spy(texts):
        seen.extend(texts)
        return [_embed(t) for t in texts]

    kb = KnowledgeBase(KNOWLEDGE_DIR, embed_fn=spy, embed_model="nomic-embed-text")
    assert all(t.startswith("search_document: ") for t in seen)
    seen.clear()
    kb.search("petals", k=1)
    assert seen == ["search_query: petals"]

    seen.clear()
    KnowledgeBase(KNOWLEDGE_DIR, embed_fn=spy, embed_model="mxbai-embed-large")
    assert not any(t.startswith("search_") for t in seen)
