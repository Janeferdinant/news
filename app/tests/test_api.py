"""API tests with mocked models (no artifacts, torch or NLTK data needed)."""
import pytest
from server import DEFAULT_MAX_CHARS, create_app
from services import Registry


class FakeModel:
    """Stands in for a real model; records what it was asked."""

    def __init__(self, model_id, name, label="fake", confidence=0.9, terms=True, fail=False):
        self.id, self.name, self.label, self.confidence = model_id, name, label, confidence
        self.terms, self.fail, self.calls = terms, fail, []

    def info(self):
        return {"id": self.id, "name": self.name, "available": True, "kind": "mock",
                "description": "mock", "explains": self.terms}

    def predict(self, title, text):
        self.calls.append((title, text))
        if self.fail:
            raise RuntimeError("boom")
        other = round(1 - self.confidence, 4)
        fake_p, real_p = (self.confidence, other) if self.label == "fake" else (other, self.confidence)
        probs = {"fake": fake_p, "real": real_p}
        return {"model": self.id, "name": self.name, "label": self.label, "confidence": self.confidence,
                "probabilities": probs,
                "top_terms": [{"term": "shocking", "contribution": -1.2, "direction": "fake", "value": 0.2}]
                if self.terms else None}


@pytest.fixture
def models():
    return {
        "logistic_regression": FakeModel("logistic_regression", "Logistic Regression", "fake", 0.91),
        "xgboost": FakeModel("xgboost", "XGBoost", "real", 0.77),
        "distilbert": FakeModel("distilbert", "DistilBERT", "fake", 0.99, terms=False),
    }


@pytest.fixture
def client(models):
    app = create_app(registry=Registry(models), max_chars=200)
    return app.test_client()


def post(client, payload, **kw):
    return client.post("/api/predict", json=payload, **kw)


def test_health_lists_model_availability(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json() == {
        "status": "ok",
        "models": {"logistic_regression": True, "xgboost": True, "distilbert": True},
    }


def test_models_endpoint_has_three_models_no_metrics(client):
    data = client.get("/api/models").get_json()
    assert [m["id"] for m in data["models"]] == ["logistic_regression", "xgboost", "distilbert"]
    assert data["max_chars"] == 200
    assert "synthetic" in data["notice"].lower()
    assert not any("accuracy" in m for m in data["models"])  # demos never show headline metrics


def test_predict_returns_every_model(client):
    r = post(client, {"title": "A title", "text": "Some article text"})
    assert r.status_code == 200
    body = r.get_json()
    by_id = {x["model"]: x for x in body["results"]}
    assert list(by_id) == ["logistic_regression", "xgboost", "distilbert"]
    assert by_id["logistic_regression"]["label"] == "fake"
    assert by_id["xgboost"]["label"] == "real" and by_id["xgboost"]["confidence"] == 0.77
    assert by_id["logistic_regression"]["top_terms"][0]["term"] == "shocking"
    assert by_id["distilbert"]["top_terms"] is None
    assert body["unavailable"] == []
    assert "synthetic" in body["notice"].lower()


def test_predict_strips_whitespace_and_passes_title(client, models):
    post(client, {"title": "  Hello  ", "text": "  body  "})
    assert models["xgboost"].calls == [("Hello", "body")]


def test_title_is_optional(client, models):
    assert post(client, {"text": "body"}).status_code == 200
    assert models["xgboost"].calls[-1] == ("", "body")


def test_models_filter(client, models):
    body = post(client, {"text": "body", "models": ["xgboost"]}).get_json()
    assert [x["model"] for x in body["results"]] == ["xgboost"]
    assert models["distilbert"].calls == []


@pytest.mark.parametrize("payload,code", [
    ({}, "invalid_text"),
    ({"text": 5}, "invalid_text"),
    ({"text": None}, "invalid_text"),
    ({"text": "   "}, "empty_text"),
    ({"text": "ok", "title": 7}, "invalid_title"),
    ({"text": "ok", "models": "xgboost"}, "invalid_models"),
    ({"text": "ok", "models": []}, "invalid_models"),
    ({"text": "ok", "models": ["gpt"]}, "unknown_model"),
])
def test_validation_errors_are_400(client, payload, code):
    r = post(client, payload)
    assert r.status_code == 400
    assert r.get_json()["code"] == code


def test_non_object_json_is_400(client):
    r = client.post("/api/predict", json=["text"])
    assert r.status_code == 400 and r.get_json()["code"] == "invalid_json"


def test_malformed_json_is_400(client):
    r = client.post("/api/predict", data="{not json", content_type="application/json")
    assert r.status_code == 400


def test_wrong_content_type_is_415(client):
    r = client.post("/api/predict", data="text=hello", content_type="text/plain")
    assert r.status_code == 415


def test_length_cap_boundary(client):
    assert post(client, {"text": "a" * 200}).status_code == 200
    r = post(client, {"text": "a" * 201})
    assert r.status_code == 413 and r.get_json()["code"] == "text_too_long"


def test_title_length_cap(client):
    r = post(client, {"text": "ok", "title": "t" * 301})
    assert r.status_code == 413 and r.get_json()["code"] == "title_too_long"


def test_huge_body_is_rejected_before_parsing(client):
    r = client.post("/api/predict", data=b'{"text": "' + b"a" * 100_000 + b'"}', content_type="application/json")
    assert r.status_code == 413


def test_default_cap_is_documented_value():
    assert DEFAULT_MAX_CHARS == 6000


def test_missing_model_is_reported_not_fatal(models):
    del models["distilbert"]
    reg = Registry(models, missing={"distilbert": "Could not load DistilBERT from distilbert/: OSError"})
    c = create_app(registry=reg, max_chars=200).test_client()
    listing = {m["id"]: m for m in c.get("/api/models").get_json()["models"]}
    assert listing["distilbert"]["available"] is False
    assert c.get("/api/health").get_json()["models"]["distilbert"] is False
    body = post(c, {"text": "body"}).get_json()
    assert len(body["results"]) == 2
    assert body["unavailable"][0]["model"] == "distilbert"
    assert "OSError" in body["unavailable"][0]["reason"]


def test_one_failing_model_does_not_hide_the_others(models):
    models["xgboost"].fail = True
    c = create_app(registry=Registry(models), max_chars=200).test_client()
    body = post(c, {"text": "body"}).get_json()
    assert [x["model"] for x in body["results"]] == ["logistic_regression", "distilbert"]
    assert body["unavailable"] == [{"model": "xgboost", "reason": "Prediction failed; see the server log."}]
    assert "boom" not in str(body)  # internal errors are not leaked


def test_no_models_available_is_503():
    c = create_app(registry=Registry({}, missing={}), max_chars=200).test_client()
    r = post(c, {"text": "body"})
    assert r.status_code == 503 and r.get_json()["code"] == "no_models"


def test_unknown_route_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["code"] == "not_found"


def test_get_on_predict_is_405(client):
    assert client.get("/api/predict").status_code == 405
