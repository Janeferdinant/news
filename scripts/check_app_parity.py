"""End-to-end check of the app's single-article pipeline against the notebook's test accuracy.

Scores a seeded sample of the notebook's held-out test articles one at a time through the same code
the Flask API uses (fakenews.features + app/services.py) and prints the sample accuracy of each
classical model. It should be close to the notebook's full-test-split accuracy; it is a sanity
check on the app, not a new benchmark.

    python scripts/check_app_parity.py --models outputs/models --n 400
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from services import ClassicalModels  # noqa: E402

from fakenews.data import LABEL_NAMES, SEED, load_dataframe, notebook_split  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=str(ROOT / "outputs" / "models"))
    p.add_argument("--data", default=str(ROOT / "data" / "fake_news_dataset.csv"))
    p.add_argument("--n", type=int, default=400)
    args = p.parse_args()

    df = load_dataframe(args.data)
    _, test_idx = notebook_split(df)
    sample = random.Random(SEED).sample(list(test_idx), args.n)
    service = ClassicalModels(args.models)

    correct = {"logistic_regression": 0, "xgboost": 0}
    for i in sample:
        row = df.loc[i]
        out = service.predict_both(row["title"], row["text"])
        for key in correct:
            correct[key] += out[key]["label"] == row["label"]
    print(f"Single-article pipeline on {args.n} test articles (labels {LABEL_NAMES}):")
    for key, n_ok in correct.items():
        print(f"  {key}: {n_ok / args.n:.4f}")


if __name__ == "__main__":
    main()
