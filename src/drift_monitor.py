"""
Data & prediction drift monitoring using Evidently AI.

Compares a "reference" batch (the distribution the model was trained/
validated on) against a "current" batch (recent production traffic) and
produces an HTML report plus a machine-readable JSON summary. In the
CI/CD pipeline this is what would run on a schedule (e.g. nightly)
against fresh production logs — see the drift-check job in
.github/workflows/ci-cd.yml.

We monitor two signals a text classifier commonly drifts on:
  1. Text-level features (length, word count) — cheap proxies for
     "the kind of thing users are writing has changed."
  2. Predicted label distribution — if the model suddenly starts
     predicting "technical" for 70% of traffic when it used to be 25%,
     that's worth a human looking at even before accuracy can be
     measured (true labels usually lag production by days).

Usage:
    python -m src.drift_monitor \\
        --reference data/tickets_reference.csv \\
        --current data/tickets_current.csv \\
        --out-dir artifacts/drift_report
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data import clean_text
from src.model import TicketClassifier


def _add_text_features(df: pd.DataFrame, clf: TicketClassifier | None = None) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().apply(len)
    if clf is not None:
        df["predicted_label"] = df["text"].apply(lambda t: clf.predict(t).label)
        df["prediction_confidence"] = df["text"].apply(lambda t: clf.predict(t).confidence)
    return df


def run_drift_check(reference_path: str, current_path: str, out_dir: str,
                     model_path: str | None = None) -> dict:
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    clf = TicketClassifier.from_checkpoint(model_path) if model_path else None

    reference = _add_text_features(reference_df, clf)
    current = _add_text_features(current_df, clf)

    numeric_cols = ["text_length", "word_count"]
    if clf is not None:
        numeric_cols.append("prediction_confidence")
    categorical_cols = ["label"] + (["predicted_label"] if clf is not None else [])

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        reference_data=reference[numeric_cols + categorical_cols],
        current_data=current[numeric_cols + categorical_cols],
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out_dir / "drift_report.html"))

    result_dict = result.dict()
    summary = _summarize(result_dict, reference, current, categorical_cols)
    with open(out_dir / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _summarize(result_dict: dict, reference: pd.DataFrame, current: pd.DataFrame,
               categorical_cols) -> dict:
    """Pulls out the handful of numbers a CI job or an on-call engineer
    actually wants to see, rather than the full nested Evidently JSON."""
    share_drifted = None
    try:
        for metric in result_dict.get("metrics", []):
            m_name = metric.get("metric_name", "")
            if "DriftedColumnsCount" in m_name:
                value = metric.get("value", {})
                share_drifted = value.get("share")
    except (KeyError, TypeError, AttributeError) as e:
        print(f"Warning: could not extract drift share from Evidently result: {e}")

    label_dist_reference = reference["label"].value_counts(normalize=True).round(3).to_dict()
    label_dist_current = current["label"].value_counts(normalize=True).round(3).to_dict()

    return {
        "share_of_drifted_columns": share_drifted,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "label_distribution_reference": label_dist_reference,
        "label_distribution_current": label_dist_current,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/tickets_reference.csv")
    parser.add_argument("--current", default="data/tickets_current.csv")
    parser.add_argument("--out-dir", default="artifacts/drift_report")
    parser.add_argument("--model-path", default=None,
                         help="Optional: include prediction drift, not just data drift")
    parser.add_argument("--fail-above", type=float, default=None,
                         help="Exit non-zero if share_of_drifted_columns exceeds this (CI gate)")
    args = parser.parse_args()

    summary = run_drift_check(args.reference, args.current, args.out_dir, args.model_path)
    print(json.dumps(summary, indent=2))

    if (args.fail_above is not None and summary["share_of_drifted_columns"] is not None
            and summary["share_of_drifted_columns"] > args.fail_above):
        print(f"\nFAIL: {summary['share_of_drifted_columns']:.0%} of columns drifted "
              f"(threshold {args.fail_above:.0%})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
