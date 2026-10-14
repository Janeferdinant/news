# Third-party notices

Original work in this repository is © Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`). No third-party code or data is copied into this repository. The components below are loaded or installed at run time and keep their own licenses.

| Component | Used for | Source | License |
|---|---|---|---|
| React and ReactDOM 18.2.0 | `index.html`, loaded from a CDN | https://cdn.jsdelivr.net/npm/react@18.2.0/ and `react-dom@18.2.0` | MIT (Copyright (c) Meta Platforms, Inc. and affiliates) |
| plotly.js 2.27.0 | charts in `index.html`, loaded from a CDN | https://cdn.plot.ly/plotly-2.27.0.min.js | MIT (Copyright (c) Plotly, Inc.) |
| @babel/standalone 7.22.9 | optional: compiles `assets/app.jsx` to `assets/app.js` (`scripts/build_jsx.mjs`); installed with npm, not copied here | https://www.npmjs.com/package/@babel/standalone | MIT |
| Python packages in `requirements.txt` (scikit-learn, XGBoost, pandas, NumPy, NLTK, TextBlob, WordCloud, Matplotlib, Plotly, joblib, SciPy) | notebook | PyPI | permissive licenses (BSD-3-Clause, MIT, Apache-2.0 and similar), installed from PyPI |
| NLTK data (`punkt`, `stopwords`) | notebook preprocessing | downloaded by the notebook at run time | see https://www.nltk.org/nltk_data/ |
| Flask 3.0.3, Gunicorn 23.0.0 | the local API in `app/` (and `scripts/serve_outputs.py`) | PyPI | BSD-3-Clause (Flask), MIT (Gunicorn) |
| PyTorch 2.5.1 (CPU build) | DistilBERT fine-tuning and inference | https://download.pytorch.org/whl/cpu and PyPI | BSD-3-Clause |
| Hugging Face Transformers 4.46.3 | DistilBERT fine-tuning and inference | PyPI | Apache-2.0 |
| `distilbert-base-uncased` (model weights and tokenizer) | starting point for fine-tuning; downloaded at run time, not copied here. The fine-tuned weights are not committed. | https://huggingface.co/distilbert/distilbert-base-uncased | Apache-2.0 (Sanh et al., 2019) |
| React 18.3.1, Vite 5.4.11, Vitest 2.1.8, Testing Library, jsdom (versions pinned in `app/frontend/package.json`) | local front end and its tests, installed with npm, not copied here | https://www.npmjs.com/ | MIT (jsdom and others as stated in each package) |
| pytest, ruff | tests and lint | PyPI | MIT |

`assets/app.js` is compiled output of `assets/app.jsx` (the owner's own code) and contains no third-party code beyond what Babel's React transform inserts, which is none.

The synthetic dataset produced by `data/generate_synthetic.py` is original work. Its outlet and author names are invented; any resemblance to real ones is accidental.
