# Run log 2026-10-14: synthetic data in week1_day5_ai_interventions

Python 3.10.6 for the generator (standard library only); Python 3.10 venv with pandas 2.3.3, matplotlib 3.10.9 and nbclient 0.11.0 for the notebook. Playwright 1.59.1 (Chromium).

## What changed
All marks, groups and student ids in the folder were replaced with seeded synthetic values.

- `python week1_day5_ai_interventions/regenerate_marks.py` (seed 42, 10 students per group) wrote `intervention_data.csv` (20 rows), the chart data in `assets/plot.js` and the summary sentence in `report.html`.
- Printed: `control: n=10 mean_gain=2.00 sd=2.49`, `ai: n=10 mean_gain=4.30 sd=2.63`.
- Determinism: the script was run twice; SHA-256 of `intervention_data.csv`, `assets/plot.js` and `report.html` were identical.
- `intervention_analysis.ipynb` is now a valid notebook (it was plain Python text with an `.ipynb` name). Same analysis, executed from the folder; it writes `assets/gain_boxplot.png`, which `report.html` already referenced but did not exist.
- `report.html`: synthetic-data banner, `viewport` meta, Plotly pinned to 2.27.0 with the same SRI hash as the main page (the old `plotly-latest` tag sat after `</html>`).
- Folder README and main README state that the data is synthetic.

## Playwright on `report.html` (served locally), 390 px and 1280 px
```
PASS 390px img=true boxes=2 legend=2 overflow=0 banner=1 errors=[]
PASS 1280px img=true boxes=2 legend=2 overflow=0 banner=1 errors=[]
```

## Not changed
The original values remain in git history (commit b406436 and the merge commits). Removing them needs a history rewrite, which needs the owner's approval.
