> **Note (2026-10-13).** The original `fake_news_dataset.csv` described below is not in this repository, and nobody remembers where it came from. `data/generate_synthetic.py` was added on 2026-10-13 to regenerate a dataset with the same seven columns from invented template text. The numbers below (20,000 rows, about 5% missing, balanced labels) are what the generator produces. The original description gave real outlet names (BBC, CNN) as examples of the `source` column; the generated data uses only fictional outlets and author names. Everything in the generated data is synthetic.

Fake News Dataset Description
Overview

Filename: fake_news_dataset.csv
Size: 20,000 rows × 7 columns
Purpose: Practice fake news detection using ML and NLP
Missing Data: ~5% missing values in source and author
Labels: Balanced (real/fake)

Columns

title: Article headline (~6 words)
text: Article body (200–300 words)
date: Publication date (past 3 years)
source: Media source (e.g., BBC, CNN; ~5% missing)
author: Author's name (~5% missing)
category: Category (e.g., Politics, Health)
label: Target (real or fake)

Usage

Source: Upload to Kaggle (/kaggle/input/fake-news-dataset/fake_news_dataset.csv)
Access: Download from Kaggle or repository (if included)
Note: Synthetic data; validate before production use

