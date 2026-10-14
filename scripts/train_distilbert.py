"""Fine-tune distilbert-base-uncased on the synthetic dataset (CPU, time-boxed).

Uses the same train/test split as the notebook (fakenews.data.notebook_split), so the test
articles are never seen in training. Because full fine-tuning on 16,000 articles does not fit a
20-minute CPU budget, it trains on a documented, seeded, stratified SUBSET of the training split
for a fixed number of epochs, with a maximum token length. The held-out test split is scored once.

Results on this synthetic data say nothing about real-world fake-news detection.

Usage (from the repository root; set HF_HOME to keep the model cache outside the repo):
    python scripts/train_distilbert.py --model-dir <where to save weights>

Writes outputs/distilbert_metrics.json (committed) and saves the weights to --model-dir (not committed).
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fakenews.data import LABEL_NAMES, SEED, load_dataframe, notebook_split  # noqa: E402

MODEL_NAME = "distilbert-base-uncased"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "fake_news_dataset.csv"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "distilbert_metrics.json"))
    p.add_argument("--model-dir", required=True, help="where to save the fine-tuned weights (keep outside the repo)")
    p.add_argument("--train-subset", type=int, default=4000, help="stratified training articles used")
    p.add_argument("--val-subset", type=int, default=400, help="training-split articles held out to monitor each epoch")
    p.add_argument("--max-len", type=int, default=128, help="max tokens per article (the rest is cut off)")
    p.add_argument("--epochs", type=int, default=2, help="fixed number of epochs (no early stopping)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = library default)")
    p.add_argument("--time-budget", type=float, default=900.0, help="stop training after this many seconds")
    p.add_argument("--test-subset", type=int, default=0, help="debug only: score a sample of the test split")
    p.add_argument("--debug-rows", type=int, default=0, help="debug only: use the first N rows of the CSV")
    return p.parse_args()


def encode(tokenizer, texts, max_len):
    return tokenizer(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")


def predict(model, enc, batch_size, torch):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, enc["input_ids"].shape[0], batch_size):
            batch = {k: v[i:i + batch_size] for k, v in enc.items()}
            logits.append(model(**batch).logits)
    return torch.cat(logits).numpy()


def main():
    args = parse_args()

    import torch
    import transformers
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    df = load_dataframe(args.data)
    if args.debug_rows:
        df = df.iloc[:args.debug_rows].copy()
    train_idx, test_idx = notebook_split(df, args.seed)
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test (same call as the notebook)")

    # Documented subset of the TRAINING split only: stratified, seeded. The test split is untouched.
    y_all = df["label_encoded"]
    rest_idx, fit_idx = train_test_split(train_idx, test_size=args.train_subset, random_state=args.seed,
                                         stratify=y_all.loc[train_idx])
    _, val_idx = train_test_split(rest_idx, test_size=args.val_subset, random_state=args.seed,
                                  stratify=y_all.loc[rest_idx])
    assert not (set(fit_idx) | set(val_idx)) & set(test_idx), "leakage: training rows overlap the test split"
    if args.test_subset:
        test_idx = test_idx[:args.test_subset]
    print(f"Using {len(fit_idx)} training articles, {len(val_idx)} validation articles "
          f"(from the training split), {len(test_idx)} test articles")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "fake", 1: "real"}, label2id={"fake": 0, "real": 1})

    t0 = time.perf_counter()
    enc_fit = encode(tokenizer, df.loc[fit_idx, "content"], args.max_len)
    enc_val = encode(tokenizer, df.loc[val_idx, "content"], args.max_len)
    y_fit = torch.tensor(y_all.loc[fit_idx].values, dtype=torch.long)
    y_val = y_all.loc[val_idx].values
    tokenise_seconds = time.perf_counter() - t0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n = len(fit_idx)
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / total_steps)))

    gen = torch.Generator().manual_seed(args.seed)
    epoch_log, stopped_early, step = [], False, 0
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n, generator=gen)
        running = 0.0
        batches_done = 0
        for b in range(steps_per_epoch):
            ids = order[b * args.batch_size:(b + 1) * args.batch_size]
            out = model(input_ids=enc_fit["input_ids"][ids], attention_mask=enc_fit["attention_mask"][ids],
                        labels=y_fit[ids])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += out.loss.item()
            batches_done += 1
            step += 1
            if step % 25 == 0:
                print(f"  step {step}/{total_steps} loss {running / batches_done:.4f} "
                      f"elapsed {time.perf_counter() - train_start:.0f}s", flush=True)
            if time.perf_counter() - train_start > args.time_budget:
                stopped_early = True
                break
        val_pred = predict(model, enc_val, 64, torch).argmax(1)
        val_acc = float(accuracy_score(y_val, val_pred))
        epoch_log.append({"epoch": epoch + 1, "steps": step, "train_loss": round(running / max(1, batches_done), 4),
                          "validation_accuracy": round(val_acc, 4)})
        print(f"Epoch {epoch + 1}: train loss {epoch_log[-1]['train_loss']}, validation accuracy {val_acc:.4f}")
        if stopped_early:
            print("Time budget reached; stopping training early.")
            break
    train_seconds = time.perf_counter() - train_start

    # Score the held-out test split once (tokenisation is counted in the inference time).
    t0 = time.perf_counter()
    enc_test = encode(tokenizer, df.loc[test_idx, "content"], args.max_len)
    logits = predict(model, enc_test, 64, torch)
    inference_seconds = time.perf_counter() - t0
    y_test = y_all.loc[test_idx].values
    pred = logits.argmax(1)

    p_c, r_c, f_c, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1], zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    metrics = {
        "model": "DistilBERT",
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_macro": round(float(p_m), 4), "recall_macro": round(float(r_m), 4), "f1_macro": round(float(f_m), 4),
        "per_class": {LABEL_NAMES[i]: {"precision": round(float(p_c[i]), 4), "recall": round(float(r_c[i]), 4),
                                       "f1": round(float(f_c[i]), 4)} for i in range(2)},
        "train_seconds": round(train_seconds, 1),
        "inference_seconds": round(inference_seconds, 1),
        "inference_ms_per_article": round(1000 * inference_seconds / len(test_idx), 2),
        "tokenise_train_seconds": round(tokenise_seconds, 1),
        "test_articles": int(len(test_idx)),
    }
    config = {
        "base_model": MODEL_NAME, "seed": args.seed, "train_subset": int(len(fit_idx)),
        "train_split_size": int(len(train_idx)), "validation_subset": int(len(val_idx)),
        "max_len": args.max_len, "epochs_planned": args.epochs, "epochs_run": len(epoch_log),
        "batch_size": args.batch_size, "learning_rate": args.lr, "optimizer": "AdamW, weight decay 0.01",
        "schedule": "10% linear warm-up then linear decay", "early_stopping": "none (fixed epochs)",
        "stopped_by_time_budget": stopped_early, "time_budget_seconds": args.time_budget,
        "split": "train_test_split(df.index, test_size=0.2, random_state=42, stratify=label) as in the notebook",
        "input": "title + ' ' + text, truncated to max_len tokens",
        "epoch_log": epoch_log,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cpu_threads": torch.get_num_threads(),
                        "logical_cpus": os.cpu_count(), "platform": platform.system() + " " + platform.release()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"metrics": metrics, "config": config}, indent=2) + "\n")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(model_dir / "training_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(config, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print("Saved weights to", model_dir)


if __name__ == "__main__":
    main()
