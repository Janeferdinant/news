# Run log 2026-10-14: DistilBERT added to the comparison; Flask + React app checked

Status: **Verified (re-run 2026-10-14 on regenerated synthetic data, seed=42, subset=1,600 of 16,000 training articles, max_len=128)**.

> **These numbers describe synthetic, template-generated articles. They say nothing about real-world fake-news detection.** High scores on synthetic data are expected, and a near-perfect score would not be evidence of anything. Here all three models land between 0.84 and 0.85 because `data/generate_synthetic.py` deliberately writes about 15% of the articles of each label in the other label's style; the models are separating a writing-style recipe, not true from false. These runs have no relation to any earlier, unrecorded result on the original dataset, which is missing from this repository.

## Environment

Python 3.11.9, Windows 11, 16 logical CPUs (PyTorch used 8 threads), 32 GB RAM, **CPU only**. Fresh venv outside the repository (`requirements-dev.txt`): torch 2.5.1+cpu (installed from `https://download.pytorch.org/whl/cpu`), transformers 4.46.3, numpy 1.24.3, scikit-learn 1.3.0, xgboost 1.7.6, flask 3.0.3, pytest 8.3.3, ruff 0.7.4. The Hugging Face cache (`HF_HOME`) and the saved weights are outside the repository. The machine was shared with other jobs that used most of the CPU during these runs, so **the times below are noisy and are not a benchmark**.

## Data and split

`python data/generate_synthetic.py` output, SHA-256 `bd1d81f9236c8ee554d32c2053f5e1f0529cfb5cc7d657deef70eb1a53b4a923` (unchanged from the 2026-10-13 run). The split is the notebook's: `train_test_split(df.index, test_size=0.2, random_state=42, stratify=label)`, giving 16,000 train and 4,000 test articles (`fakenews/data.py` repeats the call; the DistilBERT script asserts that no training or validation row is in the test split). The test split was scored once.

## DistilBERT

```bash
HF_HOME=<cache outside the repo> python scripts/train_distilbert.py --train-subset 1600 --val-subset 400 --max-len 128 --epochs 2 --batch-size 16 --time-budget 700 --model-dir <weights folder outside the repo>
```

- Model: `distilbert-base-uncased`, sequence classification head, labels `fake`=0, `real`=1.
- Subset: a seeded (42), stratified sample of **1,600 of the 16,000 training articles** (10%), plus 400 other training articles held out to watch validation accuracy. Full fine-tuning does not fit a 20-minute CPU budget.
- Input: title + " " + text, truncated to **128 tokens** (articles are 200 to 320 words, so most of each article is not read).
- Optimiser: AdamW, learning rate 5e-5, weight decay 0.01, batch size 16, 10% warm-up then linear decay, gradient clipping 1.0. Fixed epoch count (2); no early stopping.
- **The 700-second training time budget stopped training at step 156 of 200** (1.56 epochs), because the shared CPU made steps slower than in a trial run. The learning-rate schedule had therefore not finished decaying. The saved model is the one at that point.
- Validation accuracy (400 training-split articles): 0.8725 after epoch 1, 0.8700 at the stop.
- Total wall time of the script: 1,020 s (17 minutes), inside the 20-minute limit.

| Metric (4,000 test articles) | DistilBERT |
|---|---|
| Accuracy | 0.8532 |
| Precision, macro | 0.8533 |
| Recall, macro | 0.8533 |
| F1, macro | 0.8532 |
| Per class (fake / real) precision | 0.8563 / 0.8503 |
| Per class recall | 0.8490 / 0.8575 |
| Per class F1 | 0.8526 / 0.8539 |
| Training time | 726.8 s (156 steps) |
| Inference time, 4,000 articles incl. tokenisation | 256.6 s (64 ms per article) |

## Classical models (notebook re-executed)

`jupyter nbconvert --to notebook --execute --inplace fake_news_detection_enhanced_kaggle.ipynb` was run twice in this session (the notebook now also records precision, recall, F1 and times; 311 s for the second run, slower than 141 s on 2026-10-13 because of the shared CPU). `outputs/mi_scores.json` and `outputs/selected_features.txt` have the same SHA-256 as on 2026-10-13 (`788c12b2...`, `9638db69...`), so the selection and the seeding are unchanged. The final notebook cell calls `scripts/build_comparison.py`, which merges `outputs/distilbert_metrics.json` and writes `outputs/model_metrics.json` and `outputs/model_comparison.json`.

| Metric (4,000 test articles) | Logistic Regression | XGBoost |
|---|---|---|
| Accuracy | 0.8400 | 0.8385 |
| Precision, macro | 0.8403 | 0.8385 |
| Recall, macro | 0.8400 | 0.8385 |
| F1, macro | 0.8400 | 0.8385 |
| Training time (192 selected features, model only) | 0.2 s | 6.8 s |
| Inference time, test split (features already computed) | 1.8 ms | 8.5 ms |

Feature selection (clustering-based MI plus ACO) took 125.8 s. The three models differ by at most 1.5 points on one split and one seed, so no ranking should be drawn from them. The classical inference times exclude text cleaning and TF-IDF; the DistilBERT time includes tokenisation, so the two are not like for like.

## Flask + React app

- `pytest -q`: 34 passed (API with mocked models; classical service with tiny real models). `ruff check .`: all checks passed.
- `npx vitest run` in `app/frontend`: 7 passed.
- The API was started locally with the real artifacts (`python app/server.py`, loopback only): `/api/health` reported all three models loaded; `/api/predict` returned label, confidence and top terms for Logistic Regression and XGBoost, and label and confidence for DistilBERT; a 7,000-character text got HTTP 413.
- `python scripts/check_app_parity.py --n 400` (single-article pipeline on 400 seeded test articles): Logistic Regression 0.8250, XGBoost 0.8225. These are within sampling error of the full-split 0.8400 and 0.8385 (about 1.8 points standard error at n=400), which is the check that the app's preprocessing matches the notebook's.
- Playwright (Chromium) on the built app served by Flask at 390 px and 1280 px, "Sensational style example" then Analyse: the banner, three result cards (label, confidence, top terms; DistilBERT states that it has no term-level explanation), no horizontal overflow (0 px), no console errors. Screenshots: `docs/app-390.png`, `docs/app-1280.png`.
- Playwright on the static page `index.html` (served with `python -m http.server`) at 390 px and 1280 px: the metrics table with three models, the accuracy chart with four bars (baseline and three models), the banner, the "static results view" note, 0 px overflow. The only console message was the usual missing `favicon.ico` (404).

Model weights are not committed. They were saved to `C:\...\portfolio-audit\artifacts\news\distilbert` (about 268 MB), with a copy of the notebook's pickles in `artifacts\news\classical`.

## Fresh-clone check of the README commands

The branch was cloned into a new folder and the README's "Getting started" commands were run in a new Python 3.11.9 virtual environment (`requirements.txt`, then `requirements-dev.txt` with the CPU PyTorch index), with the Hugging Face cache outside the clone:

- Step 1: the generator printed the SHA-256 above; the notebook executed in 3.6 minutes. `mi_scores.json` and `selected_features.txt` were identical (same SHA-256); only the timings in `classical_metrics.json` and `model_metrics.json` changed, as expected.
- Step 2: `python scripts/train_distilbert.py ...` with the README's arguments and `python scripts/build_comparison.py` ran without error. This re-run had less CPU contention, so **it completed all 200 steps (2 full epochs) without the time guard**: test accuracy 0.8522 (precision 0.8523, recall 0.8522, F1 0.8522), validation accuracy 0.8725 after both epochs, training 666 s, inference 127 s. The step-125 and step-150 losses of the first 100 steps matched the logged run exactly (the seeding is deterministic), and the 0.1-point difference in accuracy (0.8532 logged, stopped at step 156, versus 0.8522 here) comes from where the time guard stopped the first run. This re-run was **not committed**; the committed numbers are the first run's.
- Step 3: `npm ci`, `npm run build` and `npm test` (7 passed), `pytest -q` (34 passed) and `ruff check .` (all checks passed) succeeded; `python app/server.py` served the built page and `/api/predict` returned all three models; `python scripts/check_app_parity.py --n 100` gave 0.88 (Logistic Regression) and 0.89 (XGBoost) on 100 articles.
- Docker: not built (the Docker engine was not running).
