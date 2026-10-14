"""Load the synthetic dataset and reproduce the notebook's train/test split exactly."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABELS = {"fake": 0, "real": 1}  # same encoding as the notebook: real=1, fake=0
LABEL_NAMES = ["fake", "real"]  # index = encoded label


def load_dataframe(path):
    """Read the CSV and prepare the columns used by the models (mirrors notebook section 3)."""
    df = pd.read_csv(Path(path))
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]
    df["label_encoded"] = df["label"].map(LABELS)
    return df


def notebook_split(df, seed=SEED):
    """Stratified 80/20 split of the row index, identical to the notebook (section 4).

    The notebook calls train_test_split(df.index, test_size=0.2, random_state=42,
    stratify=df['label_encoded']). Anything that learns from data must use train_idx only.
    """
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=seed, stratify=df["label_encoded"]
    )
    return train_idx, test_idx
