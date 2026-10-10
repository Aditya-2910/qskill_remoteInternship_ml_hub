"""Chat agent: a local LLM that calls the trained models as tools and answers with RAG context.

Flow for one user message
    1. Retrieve the most relevant knowledge-base chunks and put them in the system prompt (RAG).
    2. Ask the LLM. If it answers with ``tool_calls`` we validate the arguments with pydantic,
       run the real model, and feed the JSON result back as a ``tool`` message.
    3. Repeat (bounded) until the LLM produces a plain answer.

The LLM never computes a prediction itself: it only extracts the inputs and explains the result.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from mlhub.rag.store import KnowledgeBase
from mlhub.schemas import (
    ChatMessage,
    ChatResponse,
    IrisFeatures,
    SourceChunk,
    ToolCallRecord,
)
from mlhub.services import ModelNotAvailable, ModelService

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the assistant of ML Hub, a small platform with three machine-learning models: \
an iris flower classifier, an SMS spam detector and a California house price estimator.

Rules:
- To classify an iris flower, check whether a message is spam, or estimate a house price you MUST call \
the matching tool. Never invent a prediction.
- Iris needs all four measurements. If any is missing, ask the user for it instead of guessing.
- For house prices, call the tool with only the values the user gave. Missing values are filled with \
typical values, and you must tell the user which ones were assumed.
- For explanations and general questions, use the CONTEXT below. If the context does not cover the \
question, say so honestly instead of making something up.
- After a tool returns, explain the result in plain language: the prediction, the probabilities \
(if any) and, for spam, which words mattered.
- Be concise: a short paragraph, no long lists."""


class LLM(Protocol):
    model: str

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, temperature: float = 0.1
    ) -> dict[str, Any]: ...


class HousePartial(BaseModel):
    """House features as the LLM may supply them: any subset, extras ignored."""

    model_config = ConfigDict(extra="ignore")
    MedInc: float | None = None
    HouseAge: float | None = None
    AveRooms: float | None = None
    AveBedrms: float | None = None
    Population: float | None = None
    AveOccup: float | None = None
    Latitude: float | None = None
    Longitude: float | None = None


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_num = lambda desc: {"type": "number", "description": desc}  # noqa: E731

TOOLS: list[dict[str, Any]] = [
    _fn(
        "predict_iris",
        "Classify an iris flower as setosa, versicolor or virginica from its measurements in cm.",
        {
            "sepal_length": _num("Sepal length in cm"),
            "sepal_width": _num("Sepal width in cm"),
            "petal_length": _num("Petal length in cm"),
            "petal_width": _num("Petal width in cm"),
        },
        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    ),
    _fn(
        "predict_spam",
        "Decide whether a text message is spam or ham (not spam) and explain which words mattered.",
        {"text": {"type": "string", "description": "The exact message text to classify"}},
        ["text"],
    ),
    _fn(
        "estimate_house_price",
        "Estimate the median house value of a California census block group. Provide only the values "
        "the user mentioned; the rest are filled with typical values.",
        {
            "MedInc": _num("Median household income in units of $10,000 (3.5 means $35,000)"),
            "HouseAge": _num("Median house age in years"),
            "AveRooms": _num("Average number of rooms per household"),
            "AveBedrms": _num("Average number of bedrooms per household"),
            "Population": _num("Number of people in the block group"),
            "AveOccup": _num("Average number of occupants per household"),
            "Latitude": _num("Latitude (32 to 42 inside California)"),
            "Longitude": _num("Longitude (-124 to -114 inside California)"),
        },
        [],
    ),
]


class ToolRunner:
    """Validates tool arguments and runs the real models. Errors are returned, not raised,
    so the LLM can read them and ask the user for what is missing."""

    def __init__(self, models: ModelService):
        self.models = models

    def run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "predict_iris":
                pred = self.models.predict_iris(IrisFeatures(**arguments))
                return {"species": pred.prediction, "probabilities": pred.probabilities}
            if name == "predict_spam":
                text = arguments.get("text")
                if not isinstance(text, str) or not text.strip():
                    return {"error": "The 'text' argument is required."}
                pred = self.models.predict_spam(text[:5000])
                return {
                    "label": pred.prediction,
                    "probabilities": pred.probabilities,
                    "influential_words": pred.details.get("top_terms", []),
                }
            if name == "estimate_house_price":
                partial = HousePartial(**arguments).model_dump()
                pred, assumed = self.models.predict_house_partial(partial)
                return {
                    "estimated_price_usd": round(float(pred.prediction)),
                    "assumed_typical_values_for": assumed,
                    "note": "Median value for a 1990 California census block group, not a single house.",
                }
            return {"error": f"Unknown tool '{name}'."}
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return {"error": f"Invalid arguments: {problems}"}
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments: {exc}"}
        except ModelNotAvailable as exc:
            return {"error": str(exc)}


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class Agent:
    def __init__(self, llm: LLM, models: ModelService, kb: KnowledgeBase | None, max_steps: int = 4, top_k: int = 3):
        self.llm = llm
        self.tools = ToolRunner(models)
        self.kb = kb
        self.max_steps = max_steps
        self.top_k = top_k

    def _system_message(self, sources: list[SourceChunk]) -> str:
        if not sources:
            return SYSTEM_PROMPT + "\n\nCONTEXT: (nothing relevant was found in the knowledge base)"
        context = "\n\n".join(f"[{i}] ({s.source})\n{s.text}" for i, s in enumerate(sources, 1))
        return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"

    def run(self, message: str, history: list[ChatMessage] | None = None) -> ChatResponse:
        sources = self.kb.search(message, k=self.top_k) if self.kb else []
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_message(sources)}]
        messages += [{"role": m.role, "content": m.content} for m in (history or [])[-10:]]
        messages.append({"role": "user", "content": message})

        records: list[ToolCallRecord] = []
        answer = ""
        steps = 0
        for step in range(1, self.max_steps + 1):
            steps = step
            reply = self.llm.chat(messages, tools=TOOLS)
            calls = reply.get("tool_calls") or []
            if not calls:
                answer = (reply.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
            for call in calls[:4]:
                fn = call.get("function", {})
                name = fn.get("name", "")
                arguments = _parse_arguments(fn.get("arguments"))
                result = self.tools.run(name, arguments)
                records.append(ToolCallRecord(name=name, arguments=arguments, result=result))
                messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
        else:
            # Tool budget used up: force a final answer without offering tools again.
            answer = (self.llm.chat(messages).get("content") or "").strip()

        if not answer:
            answer = self._fallback_answer(records)
        return ChatResponse(answer=answer, tool_calls=records, sources=sources, llm_model=self.llm.model, steps=steps)

    @staticmethod
    def _fallback_answer(records: list[ToolCallRecord]) -> str:
        if not records:
            return "I could not produce an answer. Please try rephrasing your question."
        last = records[-1]
        return f"The model returned: {json.dumps(last.result)}"
