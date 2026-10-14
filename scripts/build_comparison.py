"""Combine the classical and DistilBERT results into the files the static page reads.

Reads   outputs/classical_metrics.json   (written by the notebook)
        outputs/distilbert_metrics.json  (written by scripts/train_distilbert.py; optional)
Writes  outputs/model_metrics.json       (table: accuracy, precision, recall, F1, times, status label)
        outputs/model_comparison.json    (Plotly figure of test accuracy, read by index.html)

The notebook calls build() at its end, so re-running the notebook keeps the DistilBERT bar.
All numbers come from runs on synthetic data and say nothing about real-world fake-news detection.
"""
import json
import sys
from pathlib import Path

import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
RUN_DATE = "2026-10-14"
CHART_TITLE = "Test accuracy on synthetic data (not a real-world result)"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row(name, key, m, status, extra):
    return {
        "id": key, "name": name, "accuracy": m["accuracy"],
        "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
        "train_seconds": m["train_seconds"], "inference_seconds": m["inference_seconds"],
        "status": status, **extra,
    }


def build(out_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "outputs"
    classical = _read(out_dir / "classical_metrics.json")
    seed = classical["seed"]
    n_train = classical["train_articles"]
    rows = []
    notes = {
        "logistic_regression": "Trained on the full training split with the ACO-selected features. "
                               "Times cover the model only (TF-IDF, mutual information and ACO are separate).",
        "xgboost": "Trained on the full training split with the ACO-selected features. "
                   "Times cover the model only (TF-IDF, mutual information and ACO are separate).",
    }
    for key, name in (("logistic_regression", "Logistic Regression"), ("xgboost", "XGBoost")):
        status = (f"Verified (re-run {RUN_DATE} on regenerated synthetic data, seed={seed}, "
                  f"subset=full training split of {n_train:,} articles)")
        rows.append(_row(name, key, classical["models"][key], status, {"note": notes[key]}))

    labels = ["Majority baseline"] + [r["name"] for r in rows]
    accuracies = [classical["baseline_accuracy"]] + [r["accuracy"] for r in rows]

    bert_path = out_dir / "distilbert_metrics.json"
    if bert_path.exists():
        bert = _read(bert_path)
        cfg, m = bert["config"], bert["metrics"]
        status = (f"Verified (re-run {RUN_DATE} on regenerated synthetic data, seed={cfg['seed']}, "
                  f"subset={cfg['train_subset']:,} of {cfg['train_split_size']:,} training articles, "
                  f"max_len={cfg['max_len']}, epochs={cfg['epochs_run']})")
        note = ("Fine-tuned distilbert-base-uncased on a subset so that it fits a CPU time budget. "
                "Inference time includes tokenisation of the test articles.")
        rows.append(_row("DistilBERT", "distilbert", m, status, {"note": note}))
        labels.append(f"DistilBERT ({cfg['train_subset']:,}-article subset)")
        accuracies.append(m["accuracy"])

    metrics = {
        "label": "Synthetic data built for teaching. These numbers say nothing about real-world fake-news detection, "
                 "and near-perfect scores on synthetic data are expected.",
        "run_date": RUN_DATE, "test_articles": classical["test_articles"],
        "baseline_accuracy": classical["baseline_accuracy"], "models": rows,
    }
    with open(out_dir / "model_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")

    fig = px.bar(x=labels, y=accuracies, title=CHART_TITLE, labels={"x": "Model", "y": "Accuracy"},
                 color_discrete_sequence=["#2563eb"])
    with open(out_dir / "model_comparison.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(fig.to_json())
    return metrics


if __name__ == "__main__":
    result = build(sys.argv[1] if len(sys.argv) > 1 else None)
    print("wrote model_metrics.json and model_comparison.json:", [r["name"] for r in result["models"]])
