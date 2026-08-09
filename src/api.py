"""
FastAPI service that serves the ticket classifier.

Model loading strategy:
  - MODEL_SOURCE=registry (default in production): loads the model
    tagged with the given alias from the MLflow Model Registry, so a
    new deployment automatically picks up whatever was last promoted
    to "production" — no manual file copying between training and
    serving.
  - MODEL_SOURCE=local: loads a plain path (what the Dockerfile bakes
    in, and what's fastest for local dev / tests).

Run locally:
    uvicorn src.api:app --reload --port 8000
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import LABELS, clean_text
from src.model import TicketClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ticket-classifier-api")

MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local")           # "local" | "registry"
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "artifacts/model")
MODEL_REGISTRY_ALIAS = os.environ.get("MODEL_REGISTRY_ALIAS", "production")
REGISTRY_MODEL_NAME = "ticket-classification"

_state = {"classifier": None, "model_version": None, "loaded_at": None}


def load_classifier() -> TicketClassifier:
    if MODEL_SOURCE == "registry":
        import mlflow.transformers
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        version = client.get_model_version_by_alias(REGISTRY_MODEL_NAME, MODEL_REGISTRY_ALIAS)
        _state["model_version"] = f"registry:v{version.version}@{MODEL_REGISTRY_ALIAS}"
        model_uri = f"models:/{REGISTRY_MODEL_NAME}@{MODEL_REGISTRY_ALIAS}"
        components = mlflow.transformers.load_model(model_uri, return_type="components")
        return TicketClassifier(components["tokenizer"], components["model"])
    else:
        _state["model_version"] = f"local:{MODEL_LOCAL_PATH}"
        return TicketClassifier.from_checkpoint(MODEL_LOCAL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model (source={MODEL_SOURCE})...")
    start = time.time()
    _state["classifier"] = load_classifier()
    _state["loaded_at"] = time.time()
    logger.info(f"Model loaded in {time.time() - start:.2f}s "
                f"(version={_state['model_version']})")
    yield
    _state["classifier"] = None


app = FastAPI(
    title="Ticket Classification API",
    description="Classifies customer support tickets into billing / technical / account / general.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                       json_schema_extra={"example": "I was charged twice for my subscription this month."})


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    labels: list


def get_classifier() -> TicketClassifier:
    """Dependency-style accessor — tests override this by monkeypatching
    src.api._state['classifier'] with a mock, so the API layer's logic
    is exercised without needing a real trained model in CI."""
    clf = _state["classifier"]
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return clf


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _state["classifier"] is not None else "loading",
        model_version=_state["model_version"],
        labels=LABELS,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    clf = get_classifier()
    text = clean_text(req.text)
    if not text:
        raise HTTPException(status_code=422, detail="text is empty after cleaning")

    start = time.time()
    result = clf.predict(text)
    latency_ms = (time.time() - start) * 1000

    logger.info(f"prediction label={result.label} confidence={result.confidence} "
                f"latency_ms={latency_ms:.1f}")

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        scores=result.scores,
        model_version=str(_state["model_version"]),
        latency_ms=round(latency_ms, 2),
    )
