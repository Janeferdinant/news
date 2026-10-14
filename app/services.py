"""Model services for the local Flask API.

Each service exposes ``info()`` and ``predict(title, text)``. The API only talks to this small
interface, so the tests replace the real models with mocks.

Classical models (Logistic Regression, XGBoost) are loaded from the pickles the notebook writes
(``fake_news_lr_model.pkl``, ``fake_news_xgb_model.pkl``, ``tfidf_vectorizer.pkl``,
``selected_indices.pkl``). DistilBERT is loaded from a ``save_pretrained`` folder written by
``scripts/train_distilbert.py``.

Everything here is trained on synthetic data built for teaching; predictions on real articles
are meaningless. Only load pickles you created yourself: unpickling runs code.
"""
import logging
import os
import threading
from pathlib import Path

import numpy as np

LABEL_NAMES = ["fake", "real"]  # index = encoded label (real=1, fake=0), as in the notebook
TOP_TERMS = 8

log = logging.getLogger(__name__)


def _label_result(model_id, name, probs, top_terms=None, extra=None):
    """Shape one model's answer. ``probs`` is [P(fake), P(real)]."""
    idx = int(np.argmax(probs))
    result = {
        "model": model_id,
        "name": name,
        "label": LABEL_NAMES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {"fake": round(float(probs[0]), 4), "real": round(float(probs[1]), 4)},
        "top_terms": top_terms,
    }
    if extra:
        result.update(extra)
    return result


def _top_terms(contrib, names, selected_values, limit=TOP_TERMS):
    """Pick the selected features that moved the score most for this article.

    ``contrib`` is a per-feature contribution in log-odds of "real" (positive pushes towards
    real, negative towards fake). Only features that occur in this article (non-zero value) are
    shown: a tree model can also react to a term being absent, but "terms in your text" is the
    explanation this demo gives.
    """
    order = np.argsort(-np.abs(contrib))
    terms = []
    for i in order:
        if len(terms) >= limit:
            break
        if contrib[i] == 0 or selected_values[i] == 0:
            continue
        terms.append({
            "term": str(names[i]),
            "contribution": round(float(contrib[i]), 4),
            "direction": "real" if contrib[i] > 0 else "fake",
            "value": round(float(selected_values[i]), 4),
        })
    return terms


class ClassicalModels:
    """TF-IDF + ACO-selected features feeding Logistic Regression and XGBoost."""

    def __init__(self, directory):
        import joblib

        from fakenews.features import feature_names

        d = Path(directory)
        self.vectorizer = joblib.load(d / "tfidf_vectorizer.pkl")
        self.selected = np.asarray(joblib.load(d / "selected_indices.pkl"))
        self.lr = joblib.load(d / "fake_news_lr_model.pkl")
        self.xgb = joblib.load(d / "fake_news_xgb_model.pkl")
        self.names = feature_names(self.vectorizer)[self.selected]

    def _features(self, title, text):
        from fakenews.features import full_feature_vector

        full = full_feature_vector(title, text, self.vectorizer)
        return full[:, self.selected]

    def predict_both(self, title, text):
        x = self._features(title, text)
        present = int(np.count_nonzero(x[0]))
        extra = {"selected_features_present": present, "selected_features_total": int(len(self.selected))}

        # Logistic Regression: contribution of feature i is coef_i * x_i (log-odds of "real").
        p_real = float(self.lr.predict_proba(x)[0, 1])
        lr_contrib = self.lr.coef_[0] * x[0]
        lr = _label_result("logistic_regression", "Logistic Regression", [1 - p_real, p_real],
                           _top_terms(lr_contrib, self.names, x[0]), extra)

        # XGBoost: exact per-feature contributions from the booster (the last column is the bias).
        import xgboost as xgb

        booster = self.xgb.get_booster()
        contrib = booster.predict(xgb.DMatrix(x, feature_names=booster.feature_names), pred_contribs=True)[0][:-1]
        p_real = float(self.xgb.predict_proba(x)[0, 1])
        xg = _label_result("xgboost", "XGBoost", [1 - p_real, p_real],
                           _top_terms(contrib, self.names, x[0]), extra)
        return {"logistic_regression": lr, "xgboost": xg}


class _ClassicalView:
    """One classical model, as a service with the common interface."""

    def __init__(self, shared, key, name, description):
        self.shared, self.key, self.name, self.description = shared, key, name, description
        self.id = key

    def info(self):
        return {"id": self.id, "name": self.name, "available": True, "kind": "classical",
                "description": self.description, "explains": True}

    def predict(self, title, text):
        return self.shared.predict_both(title, text)[self.key]


class DistilBertModel:
    """Fine-tuned DistilBERT sequence classifier (CPU). No term-level explanation is offered."""

    id = "distilbert"
    name = "DistilBERT"

    def __init__(self, directory):
        import json

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = Path(directory)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d).eval()
        cfg = d / "training_config.json"
        self.max_len = json.loads(cfg.read_text(encoding="utf-8")).get("max_len", 128) if cfg.exists() else 128
        self._lock = threading.Lock()

    def info(self):
        return {"id": self.id, "name": self.name, "available": True, "kind": "transformer",
                "description": f"distilbert-base-uncased fine-tuned on a subset of the synthetic articles; "
                               f"reads the first {self.max_len} tokens.",
                "explains": False}

    def predict(self, title, text):
        enc = self.tokenizer(title + " " + text, truncation=True, max_length=self.max_len, return_tensors="pt")
        with self._lock, self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[0].tolist()
        return _label_result(self.id, self.name, probs, None,
                             {"note": f"Only the first {self.max_len} tokens are read; no term-level explanation."})


class Registry:
    """The models the API can serve, in display order, plus the ones that could not be loaded."""

    ORDER = [
        ("logistic_regression", "Logistic Regression"),
        ("xgboost", "XGBoost"),
        ("distilbert", "DistilBERT"),
    ]

    def __init__(self, models=None, missing=None):
        self.models = dict(models or {})
        self.missing = dict(missing or {})  # id -> reason

    def listing(self):
        out = []
        for model_id, name in self.ORDER:
            if model_id in self.models:
                out.append(self.models[model_id].info())
            else:
                out.append({"id": model_id, "name": name, "available": False, "kind": None,
                            "description": self.missing.get(model_id, "Artifacts not found."), "explains": False})
        return out

    @classmethod
    def from_env(cls, env=None):
        """Load whatever artifacts exist; never fail because one model is missing.

        NEWS_CLASSICAL_DIR (default ``outputs/models``) holds the notebook's pickles and
        NEWS_DISTILBERT_DIR (default ``artifacts/distilbert``) the fine-tuned DistilBERT folder.
        """
        env = os.environ if env is None else env
        root = Path(__file__).resolve().parent.parent
        classical_dir = Path(env.get("NEWS_CLASSICAL_DIR", root / "outputs" / "models"))
        bert_dir = Path(env.get("NEWS_DISTILBERT_DIR", root / "artifacts" / "distilbert"))
        models, missing = {}, {}

        try:
            shared = ClassicalModels(classical_dir)
            models["logistic_regression"] = _ClassicalView(
                shared, "logistic_regression", "Logistic Regression",
                "Logistic Regression on the ACO-selected TF-IDF features; terms show coefficient x value.")
            models["xgboost"] = _ClassicalView(
                shared, "xgboost", "XGBoost",
                "XGBoost on the ACO-selected TF-IDF features; terms show per-feature contributions.")
        except Exception as exc:  # missing pickles, version mismatch, ...
            reason = f"Could not load the classical models from {classical_dir.name}/: {type(exc).__name__}"
            log.warning("%s (%s)", reason, exc)
            missing["logistic_regression"] = missing["xgboost"] = reason

        try:
            models["distilbert"] = DistilBertModel(bert_dir)
        except Exception as exc:
            reason = f"Could not load DistilBERT from {bert_dir.name}/: {type(exc).__name__}"
            log.warning("%s (%s)", reason, exc)
            missing["distilbert"] = reason
        return cls(models, missing)
