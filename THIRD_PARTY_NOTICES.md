# Third-party notices

Original work in this repository is © Jane Rose Ferdinant, licensed CC BY-NC-SA 4.0 (see `LICENSE`). No third-party code or data is copied into this repository. The components below are loaded or installed at run time and keep their own licenses.

| Component | Used for | Source | License |
|---|---|---|---|
| React and ReactDOM 18.2.0 | `index.html`, loaded from a CDN | https://cdn.jsdelivr.net/npm/react@18.2.0/ and `react-dom@18.2.0` | MIT (Copyright (c) Meta Platforms, Inc. and affiliates) |
| plotly.js 2.27.0 | charts in `index.html`, loaded from a CDN | https://cdn.plot.ly/plotly-2.27.0.min.js | MIT (Copyright (c) Plotly, Inc.) |
| @babel/standalone 7.22.9 | optional: compiles `assets/app.jsx` to `assets/app.js` (`scripts/build_jsx.mjs`); installed with npm, not copied here | https://www.npmjs.com/package/@babel/standalone | MIT |
| Python packages in `requirements.txt` (scikit-learn, XGBoost, pandas, NumPy, NLTK, TextBlob, WordCloud, Matplotlib, Plotly, joblib, SciPy) | notebook | PyPI | permissive licenses (BSD-3-Clause, MIT, Apache-2.0 and similar), installed from PyPI |
| NLTK data (`punkt`, `stopwords`) | notebook preprocessing | downloaded by the notebook at run time | see https://www.nltk.org/nltk_data/ |
| Flask | optional `app.py` | PyPI | BSD-3-Clause |

`assets/app.js` is compiled output of `assets/app.jsx` (the owner's own code) and contains no third-party code beyond what Babel's React transform inserts, which is none.

The synthetic dataset produced by `data/generate_synthetic.py` is original work. Its outlet and author names are invented; any resemblance to real ones is accidental.
