"""
Data loading & preprocessing shared by train.py and evaluate.py.

Kept deliberately separate from train.py so both training and batch
evaluation (and future drift jobs) use the exact same preprocessing —
a common source of training/serving skew is preprocessing logic that
drifts apart between the two.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LABELS = ["billing", "technical", "account", "general"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class Dataset:
    texts: list[str]
    labels: list[int]


def load_raw_data(csv_path: Path | None = None) -> pd.DataFrame:
    """Loads the raw labeled CSV.

    Swap this function out for a real data source in production —
    e.g. a warehouse query, an S3 object, or a labeling-tool export —
    without touching any downstream code, since everything else here
    only depends on the (text, label) DataFrame contract.
    """
    csv_path = csv_path or (DATA_DIR / "tickets_train.csv")
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"Expected columns 'text' and 'label' in {csv_path}")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Minimal, deterministic cleaning applied identically at train and
    serve time. Keep this function pure (no I/O, no randomness) — it's
    imported directly by the FastAPI service to guarantee train/serve
    parity instead of re-implementing cleaning logic twice."""
    return " ".join(text.strip().split())


def to_dataset(df: pd.DataFrame) -> Dataset:
    texts = [clean_text(t) for t in df["text"].tolist()]
    labels = [LABEL2ID[label] for label in df["label"].tolist()]
    return Dataset(texts=texts, labels=labels)


def train_val_split(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["label"]
    )


if __name__ == "__main__":
    df = load_raw_data()
    train_df, val_df = train_val_split(df)
    print(f"Loaded {len(df)} rows -> train={len(train_df)} val={len(val_df)}")
    print(df["label"].value_counts())
