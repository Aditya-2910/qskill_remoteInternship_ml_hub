"""Pydantic request/response models shared by the API, the agent and the UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------- inputs


class IrisFeatures(BaseModel):
    """Measurements of one iris flower, in centimetres."""

    sepal_length: float = Field(..., gt=0, le=15, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=15, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=15, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=15, description="Petal width in cm")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    )


class SpamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Message text to classify")

    model_config = ConfigDict(json_schema_extra={"example": {"text": "WINNER!! Claim your free prize now, reply YES"}})


class HouseFeatures(BaseModel):
    """The eight California Housing block-group features."""

    MedInc: float = Field(..., ge=0, le=100, description="Median income (units of $10,000)")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age in years")
    AveRooms: float = Field(..., gt=0, le=200, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, le=50, description="Average bedrooms per household")
    Population: float = Field(..., gt=0, le=100000, description="Block group population")
    AveOccup: float = Field(..., gt=0, le=500, description="Average occupants per household")
    Latitude: float = Field(..., ge=32, le=42.5, description="Latitude (California)")
    Longitude: float = Field(..., ge=-124.5, le=-114, description="Longitude (California)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322.0,
                "AveOccup": 2.55,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    )


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


# --------------------------------------------------------------- outputs


class Prediction(BaseModel):
    model: str
    model_version: str
    algorithm: str
    prediction: Any
    probabilities: dict[str, float] | None = None
    latency_ms: float
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class SourceChunk(BaseModel):
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    sources: list[SourceChunk] = Field(default_factory=list)
    llm_model: str
    steps: int
