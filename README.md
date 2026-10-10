# ML Hub

Three classic scikit-learn projects (**Iris classification**, **spam detection**, **house price prediction**) turned into one
small ML platform: a FastAPI backend, a Streamlit UI, experiment tracking with MLflow, prediction monitoring, and a
**local LLM assistant (Ollama) that calls the models as tools and answers questions with RAG**.

Everything runs on your own machine. There are no paid APIs and no cloud accounts.

```
                 ┌──────────────────────────── Streamlit UI (app/streamlit_app.py) ────────────────────────────┐
                 │  Assistant chat · Iris · Spam · House prices · Models · Monitoring                           │
                 └──────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                            │ HTTP
                                     ┌──────────────────────▼──────────────────────┐
                                     │           FastAPI backend (src/mlhub/api)    │
                                     │  /v1/predict/{iris,spam,house}  /v1/chat     │
                                     │  /v1/models  /v1/monitoring/*   /health      │
                                     └───────┬──────────────────┬───────────────┬───┘
                                             │                  │               │
                              ┌──────────────▼───┐   ┌──────────▼─────────┐  ┌──▼────────────────────┐
                              │ scikit-learn      │   │ Agent (tool calling)│  │ Monitoring            │
                              │ pipelines (joblib)│◄──┤ + RAG knowledge base│  │ prediction log + KS   │
                              └──────────▲────────┘   └──────────┬─────────┘  │ drift check           │
                                         │ trained by             │ Ollama     └───────────────────────┘
                              ┌──────────┴────────┐   ┌──────────▼─────────┐
                              │ training/*.py      │   │ local LLM + embeds │
                              │ + MLflow tracking  │   │ (llama3.2, nomic)  │
                              └────────────────────┘   └────────────────────┘
```

## What is in the box

| Area | What it does | Where |
|---|---|---|
| Internship task 1 | Iris: EDA plots, stratified split, 3 classifiers, CV selection, accuracy / precision / confusion matrix | `src/mlhub/training/train_iris.py` |
| Internship task 2 | Spam: cleaning, TF-IDF (lowercase, stop-words, uni+bi-grams), Naive Bayes vs Logistic Regression, F1 | `src/mlhub/training/train_spam.py` |
| Internship task 3 | House prices: EDA, imputation + scaling, Linear Regression baseline vs Ridge vs gradient boosting, MSE / RMSE / R² | `src/mlhub/training/train_house.py` |
| Backend | Pydantic-validated REST API, optional API key, health check, versioned routes | `src/mlhub/api/main.py` |
| Frontend | Streamlit app that only talks to the API (a real frontend/backend split) | `app/streamlit_app.py` |
| LLM agent | Local model chooses a tool, arguments are validated, the real model runs, the LLM explains the result | `src/mlhub/llm/agent.py` |
| RAG | Markdown knowledge base + auto-generated model cards, embeddings via Ollama, TF-IDF fallback | `src/mlhub/rag/` |
| MLOps | MLflow tracking + model registry, prediction logging, drift check, Docker Compose, GitHub Actions CI, 79 tests | see below |
| Benchmark | Classic spam model vs zero-shot LLM: quality and latency on the same test messages | `src/mlhub/benchmark.py` |

## Quick start (local)

You need **Python 3.11+**. Use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: make install

python -m mlhub.training.train_all # or: make train   (no internet? add --offline, or: make train-offline)
```

Training downloads the SMS Spam Collection (UCI) and California Housing (scikit-learn) once and caches them in `data/`.
It writes `models/`, `reports/` (plots) and `mlflow.db`. Open the experiment UI with `make mlflow` (http://localhost:5000).

Now the LLM. Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2               # about 2 GB, supports tool calling
ollama pull nomic-embed-text       # small embedding model for RAG
```

Start the backend and the UI in two terminals:

```bash
uvicorn mlhub.api.main:create_app --factory --reload     # or: make api   -> http://localhost:8000/docs
streamlit run app/streamlit_app.py                       # or: make ui    -> http://localhost:8501
```

The sidebar of the UI shows whether the API, the models and Ollama are ready. The three model pages work without Ollama;
only the Assistant needs it.

> **No `make` on Windows?** Every target is a one-liner, and the commands are shown above and in the `Makefile`.

## Docker

```bash
docker compose up --build
```

This starts Ollama (and pulls the models into a volume on first run), trains the models, then starts the API
(http://localhost:8000/docs), the UI (http://localhost:8501) and the MLflow server (http://localhost:5000). If you already
run Ollama on the host, delete the two `ollama` services from `docker-compose.yml` and set
`OLLAMA_HOST=http://host.docker.internal:11434` on the services.

## Using the API

```bash
curl -X POST localhost:8000/v1/predict/iris -H 'content-type: application/json' \
  -d '{"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.8,"petal_width":2.2}'

curl -X POST localhost:8000/v1/predict/spam -H 'content-type: application/json' \
  -d '{"text":"WINNER!! Claim your free prize now, call 07123456789"}'

curl -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"message":"Is this spam? \"URGENT! Verify your account at bit.ly/x9\""}'
```

| Route | Purpose |
|---|---|
| `GET /health` | model, Ollama and API status (public) |
| `POST /v1/predict/iris`, `/spam`, `/house` | predictions with probabilities, latency and model version |
| `POST /v1/chat` | assistant; returns the answer, the tool calls it made and the RAG sources it used |
| `GET /v1/knowledge/search?q=` | inspect what the retriever finds |
| `GET /v1/models` | metadata and metrics of every trained model |
| `GET /v1/monitoring/summary`, `/drift/{model}` | usage, latency percentiles, data drift |
| `POST /v1/admin/reload` | pick up newly trained models and rebuild the knowledge base |

Set `MLHUB_API_KEY` to require an `X-API-Key` header on every `/v1` route.

## How the assistant works

1. **RAG.** The question is embedded and compared with the knowledge-base chunks (`src/mlhub/rag/knowledge/*.md` plus a
   "model card" generated from each trained model's metrics). The best chunks go into the system prompt. If Ollama
   embeddings are unavailable the store falls back to TF-IDF, so retrieval never breaks.
2. **Tool calling.** The LLM sees three tools (`predict_iris`, `predict_spam`, `estimate_house_price`). When it asks for
   one, the arguments are validated with the same Pydantic rules as the REST API and the *real* scikit-learn model runs.
   Invalid arguments come back to the LLM as an error message so it can ask you for the missing value.
3. **Answer.** The LLM explains the tool result in plain language. The loop is capped at four steps.

The LLM never makes up a prediction, since it only extracts inputs and explains outputs. Spam text is never written to the
prediction log, only its length.

## Project layout

```
src/mlhub/
  config.py            settings from environment variables
  datasets.py          data loaders + offline stand-ins
  schemas.py           Pydantic request/response models
  services.py          model registry + predictions (+ spam word explanations)
  monitoring.py        prediction log, latency stats, KS drift check
  benchmark.py         classic model vs LLM on spam
  training/            train_iris.py  train_spam.py  train_house.py  train_all.py  common.py (MLflow, selection)
  rag/                 store.py (chunking, embeddings, search)  knowledge/*.md
  llm/                 ollama_client.py  agent.py (tools + loop)
  api/main.py          FastAPI app factory
app/streamlit_app.py   the UI
tests/                 pytest suite + fake_ollama.py (scripted stand-in for Ollama)
Dockerfile  docker-compose.yml  Makefile  .github/workflows/ci.yml
```

## Design decisions worth knowing (and explaining in an interview)

- **No leakage.** TF-IDF, imputation and scaling live inside scikit-learn `Pipeline`s. Candidate models are compared with
  cross-validation on the *training* set only, and the test set is used once, for the final numbers.
- **F1 for spam, not accuracy.** About 87 % of SMS messages are ham, so "always ham" already scores 87 % accuracy.
  Exact duplicate messages are removed before the split so the same text can't be in both train and test.
- **Housing data.** The Boston dataset was removed from scikit-learn, so this uses California Housing. Its target is capped
  at $500k and it is 1990 census data for block groups, not single houses. Serving clips predictions to the training range.
- **One saved artifact per model**: a joblib pipeline plus `metadata.json` (metrics, data source, version) and a small
  reference sample for drift checks. MLflow additionally stores every run and registers each model.
- **Honest labelling.** If a dataset can't be downloaded, small synthetic stand-ins are used and recorded as
  `data_source` in the metadata; the UI shows a warning on the Models page. Never quote those numbers as results.

## Tests and CI

```bash
pytest          # 79 tests, about 20 s, no network, no GPU, no Ollama needed
ruff check .
```

The tests train real models on offline data, run the real FastAPI app, and put a scripted fake Ollama
(`tests/fake_ollama.py`) behind it, so the agent loop, tool validation, RAG prompt building, error handling, the MLflow
registry and the Streamlit pages (through Streamlit's `AppTest`) are all exercised. GitHub Actions runs lint and tests
on Python 3.11 and 3.12, then builds the Docker image and smoke-tests it.

The fake LLM proves the plumbing, not the quality of a real model's tool use. Try the real thing with the examples on
the Assistant page.

## Spam benchmark: classic ML vs LLM

```bash
python -m mlhub.benchmark --n 100 --model llama3.2     # or: make benchmark
```

Both approaches classify the same held-out messages. You get accuracy, precision, recall, F1 and latency per message,
and the result is saved to `reports/spam_benchmark.json`. Run it with two or three different Ollama models and put the table
in your write-up. It is usually the most interesting part of the project: the small linear model is typically faster by
orders of magnitude, and the interesting question is what the LLM buys you.

## Configuration

All optional; see `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | where Ollama listens |
| `OLLAMA_MODEL` | `llama3.2` | chat model, must support tool calling |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | embedding model for RAG |
| `MLHUB_API_URL` | `http://localhost:8000` | where the UI finds the API |
| `MLHUB_API_KEY` | (empty) | if set, required as `X-API-Key` on `/v1` routes |
| `MLHUB_HOME` | current directory | where `models/`, `data/`, `logs/`, `reports/`, `mlflow.db` live |
| `MLHUB_OFFLINE` | `0` | `1` = never download data, use synthetic stand-ins |
| `MLHUB_MLFLOW` | `1` | `0` = skip MLflow tracking |

## Troubleshooting

- **"Ollama not reachable"** in the sidebar: run `ollama serve` (or start the Ollama app) and check `OLLAMA_HOST`.
- **"model ... not found"**: `ollama pull llama3.2`.
- **The assistant answers but never calls a tool**: small models are sometimes unreliable at tool calling. Try
  `qwen2.5:7b` or `llama3.1:8b` via `OLLAMA_MODEL`, and restart the API.
- **The first chat message is slow**: Ollama is loading the model into memory. Later messages are much faster.
- **Retrieval seems off with embeddings**: check `GET /v1/knowledge/search?q=...`. Scores below `MIN_SCORE` in
  `src/mlhub/rag/store.py` are dropped, and that threshold was chosen without tuning it against a real embedding model.
- **"Model 'x' has not been trained yet"**: run the training command, then `POST /v1/admin/reload` or restart the API.
- **Port already in use**: change `--port` for uvicorn, or `--server.port` for Streamlit (and set `MLHUB_API_URL`).

## Ideas for going further

1. Swap the numpy vector store for Chroma or FAISS, and add your own documents to `rag/knowledge/`.
2. Add Evidently for drift reports, or a scheduled retraining job that reloads the API.
3. Stream chat tokens (`stream: true` in Ollama) to the UI.
4. Add a fourth model, e.g. a fine-tuned transformer for spam, and put it in the benchmark.
5. Deploy: Hugging Face Spaces (Streamlit) plus a small VM or Render for the API. The LLM would need a hosted endpoint there.

## Resume bullet

> Built and containerised an end-to-end ML platform serving three scikit-learn models through a FastAPI backend and
> Streamlit UI, with a local-LLM assistant (Ollama) using tool calling and RAG; tracked experiments and registered models
> with MLflow, added prediction logging and drift checks, and covered it with 79 automated tests and GitHub Actions CI.
