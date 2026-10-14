"""Local Flask API for the synthetic fake-news teaching demo.

    GET  /api/health    liveness and which models are loaded
    GET  /api/models    the three models and whether each is available
    POST /api/predict   {"title": "...", "text": "..."} -> label, confidence and top terms per model

Run from the repository root:
    python app/server.py                   # http://127.0.0.1:5000
Environment:
    NEWS_CLASSICAL_DIR   folder with the notebook's pickles (default outputs/models)
    NEWS_DISTILBERT_DIR  folder written by scripts/train_distilbert.py (default artifacts/distilbert)
    NEWS_MAX_CHARS       longest accepted article text in characters (default 6000)
    NEWS_HOST, NEWS_PORT bind address (default 127.0.0.1:5000)

Trained on synthetic data built for teaching; not a fact-checking tool.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from services import Registry  # noqa: E402
from werkzeug.exceptions import RequestEntityTooLarge  # noqa: E402

DEFAULT_MAX_CHARS = 6000
MAX_TITLE_CHARS = 300
NOTICE = ("Trained on synthetic data built for teaching. These labels say nothing about whether a real "
          "article is true, and this is not a fact-checking tool.")
DIST = Path(__file__).resolve().parent / "frontend" / "dist"


def _error(message, status, code):
    return jsonify({"error": message, "code": code}), status


def create_app(registry=None, max_chars=None):
    """Build the Flask app. Tests pass a Registry of mock models; otherwise artifacts are loaded."""
    app = Flask(__name__, static_folder=None)
    max_chars = int(max_chars or os.environ.get("NEWS_MAX_CHARS", DEFAULT_MAX_CHARS))
    app.config["MAX_CONTENT_LENGTH"] = 4 * (max_chars + MAX_TITLE_CHARS) + 1024  # bytes: UTF-8 headroom
    reg = registry if registry is not None else Registry.from_env()
    app.config["REGISTRY"] = reg
    app.config["MAX_CHARS"] = max_chars

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "models": {m["id"]: m["available"] for m in reg.listing()},
        })

    @app.get("/api/models")
    def models():
        return jsonify({"models": reg.listing(), "max_chars": max_chars, "notice": NOTICE})

    @app.post("/api/predict")
    def predict():
        if not request.is_json:
            return _error("Send a JSON body with Content-Type: application/json.", 415, "unsupported_media_type")
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _error("The request body must be a JSON object.", 400, "invalid_json")

        text = body.get("text")
        title = body.get("title", "")
        if not isinstance(text, str):
            return _error("'text' is required and must be a string.", 400, "invalid_text")
        if not isinstance(title, str):
            return _error("'title' must be a string.", 400, "invalid_title")
        text, title = text.strip(), title.strip()
        if not text:
            return _error("'text' must not be empty.", 400, "empty_text")
        if len(text) > max_chars:
            return _error(f"'text' is {len(text)} characters; the limit is {max_chars}.", 413, "text_too_long")
        if len(title) > MAX_TITLE_CHARS:
            return _error(f"'title' is {len(title)} characters; the limit is {MAX_TITLE_CHARS}.", 413, "title_too_long")

        wanted = body.get("models")
        known = [m for m, _ in reg.ORDER]
        if wanted is not None:
            if not isinstance(wanted, list) or not all(isinstance(m, str) for m in wanted) or not wanted:
                return _error("'models' must be a non-empty list of model ids.", 400, "invalid_models")
            unknown = [m for m in wanted if m not in known]
            if unknown:
                return _error(f"Unknown model id(s): {', '.join(unknown)}.", 400, "unknown_model")
        else:
            wanted = known

        results, unavailable = [], []
        for model_id in wanted:
            model = reg.models.get(model_id)
            if model is None:
                unavailable.append({"model": model_id, "reason": reg.missing.get(model_id, "Artifacts not found.")})
                continue
            try:
                results.append(model.predict(title, text))
            except Exception:  # keep one failing model from hiding the others
                app.logger.exception("prediction failed for %s", model_id)
                unavailable.append({"model": model_id, "reason": "Prediction failed; see the server log."})
        if not results:
            return jsonify({"error": "No model is available.", "code": "no_models", "unavailable": unavailable}), 503
        return jsonify({"results": results, "unavailable": unavailable, "notice": NOTICE})

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return _error(f"The request body is too large (the text limit is {max_chars} characters).", 413,
                      "body_too_large")

    @app.errorhandler(404)
    def not_found(_):
        return _error("Not found.", 404, "not_found")

    @app.errorhandler(405)
    def not_allowed(_):
        return _error("Method not allowed.", 405, "method_not_allowed")

    # Serve the built React app when present (python app/server.py after `npm run build`).
    @app.get("/")
    def index():
        if (DIST / "index.html").exists():
            return send_from_directory(DIST, "index.html")
        return jsonify({"message": "API is running. Build the front end (cd app/frontend && npm run build) "
                                   "or start it with npm run dev.", "notice": NOTICE})

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(DIST / "assets", name)

    return app


if __name__ == "__main__":
    # Local use only: loopback by default; do not expose it through a tunnel or a public host.
    create_app().run(host=os.environ.get("NEWS_HOST", "127.0.0.1"), port=int(os.environ.get("NEWS_PORT", "5000")))
