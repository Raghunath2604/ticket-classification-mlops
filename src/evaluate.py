"""
Batch-evaluates a saved (or registered) model against a labeled CSV.

Used two ways in this project:
  1. Manually, to sanity-check a model before promoting it.
  2. By CI (see .github/workflows/ci-cd.yml) as a quality gate — the
     workflow fails the build if accuracy drops below a threshold,
     which stops a regressed model from ever reaching the registry.

Usage:
    python -m src.evaluate --model-path artifacts/model --data data/tickets_train.csv
    python -m src.evaluate --registry-alias production
"""
import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

from src.data import clean_text, load_raw_data
from src.model import TicketClassifier


def evaluate(clf: TicketClassifier, data_path: str) -> dict:
    df = load_raw_data(Path(data_path))
    texts = [clean_text(t) for t in df["text"]]
    true_labels = df["label"].tolist()

    predictions = [clf.predict(t).label for t in texts]

    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)
    return {"accuracy": accuracy, "report": report, "n_samples": len(df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                         help="Local path to a saved model (e.g. artifacts/model)")
    parser.add_argument("--registry-alias", default=None,
                         help="Load from the MLflow Model Registry by alias, e.g. 'production'")
    parser.add_argument("--data", default="data/tickets_train.csv")
    parser.add_argument("--min-accuracy", type=float, default=None,
                         help="Exit non-zero if accuracy falls below this (used as a CI gate)")
    args = parser.parse_args()

    if args.registry_alias:
        import mlflow.transformers
        model_uri = f"models:/ticket-classification@{args.registry_alias}"
        components = mlflow.transformers.load_model(model_uri, return_type="components")
        clf = TicketClassifier(components["tokenizer"], components["model"])
    elif args.model_path:
        clf = TicketClassifier.from_checkpoint(args.model_path)
    else:
        print("Provide either --model-path or --registry-alias", file=sys.stderr)
        sys.exit(2)

    results = evaluate(clf, args.data)
    print(f"n_samples={results['n_samples']}  accuracy={results['accuracy']:.4f}")
    print(json.dumps(results["report"], indent=2))

    if args.min_accuracy is not None and results["accuracy"] < args.min_accuracy:
        print(f"\nFAIL: accuracy {results['accuracy']:.4f} is below required "
              f"{args.min_accuracy:.4f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
