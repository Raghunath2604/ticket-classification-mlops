import pandas as pd
import pytest

from src.data import LABEL2ID, clean_text, to_dataset, train_val_split


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world  \n") == "hello world"


def test_clean_text_is_idempotent():
    once = clean_text("  a  b   c ")
    twice = clean_text(once)
    assert once == twice


def test_to_dataset_maps_labels_to_ids():
    df = pd.DataFrame({"text": ["a", "b"], "label": ["billing", "technical"]})
    ds = to_dataset(df)
    assert ds.labels == [LABEL2ID["billing"], LABEL2ID["technical"]]
    assert ds.texts == ["a", "b"]


def test_train_val_split_is_stratified_and_covers_all_rows():
    df = pd.DataFrame({
        "text": [f"t{i}" for i in range(40)],
        "label": (["billing", "technical", "account", "general"] * 10),
    })
    train_df, val_df = train_val_split(df, val_size=0.25)
    assert len(train_df) + len(val_df) == len(df)
    val_counts = val_df["label"].value_counts()
    assert set(val_counts.index) == {"billing", "technical", "account", "general"}


def test_load_raw_data_requires_text_and_label_columns(tmp_path):
    from src.data import load_raw_data
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        load_raw_data(bad_csv)
