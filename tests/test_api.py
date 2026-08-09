"""
Unit tests for the FastAPI service.

Deliberately does NOT load a real trained model — that would make
every CI run slow and dependent on network access to the Hugging Face
Hub. Instead we inject a lightweight fake classifier that implements
the same .predict() interface, so we're testing the API's request
handling, validation, and response shape — the actual model quality is
covered separately by src/evaluate.py in the "model-quality" CI job.
"""
import pytest
from fastapi.testclient import TestClient

from src import api
from src.model import Prediction


class FakeClassifier:
    """Implements TicketClassifier's public interface without loading
    any real weights."""

    def predict(self, text: str) -> Prediction:
        if "refund" in text.lower() or "charge" in text.lower():
            label = "billing"
        elif "crash" in text.lower() or "error" in text.lower():
            label = "technical"
        else:
            label = "general"
        scores = {label_name: 0.1 for label_name in ["billing", "technical", "account", "general"]}
        scores[label] = 0.7
        return Prediction(label=label, confidence=0.7, scores=scores)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_classifier", lambda: FakeClassifier())
    with TestClient(api.app) as c:
        yield c
    api._state["classifier"] = None


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["labels"]) == {"billing", "technical", "account", "general"}


def test_predict_billing(client):
    resp = client.post("/predict", json={"text": "I need a refund for last month's charge"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "billing"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["scores"].keys()) == {"billing", "technical", "account", "general"}
    assert body["latency_ms"] >= 0


def test_predict_technical(client):
    resp = client.post("/predict", json={"text": "The app keeps showing a crash error"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "technical"


def test_predict_rejects_empty_text(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length validation


def test_predict_rejects_missing_field(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_returns_503_when_model_not_loaded(client):
    api._state["classifier"] = None
    resp = client.post("/predict", json={"text": "anything"})
    assert resp.status_code == 503
