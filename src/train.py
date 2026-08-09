"""
Fine-tunes the ticket classifier and tracks the run in MLflow, including
registering the resulting model in the MLflow Model Registry and
promoting it between stages based on validation metrics.

Usage (real training, needs internet to pull the base model):
    python -m src.train --model-name distilbert-base-uncased --epochs 3

Usage (fast, network-free smoke test — what CI runs on every push):
    python -m src.train --offline --epochs 1
"""
import argparse
import subprocess
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.data import LABEL2ID, LABELS, clean_text, load_raw_data, train_val_split
from src.model import DEFAULT_MODEL_NAME, TicketClassifier

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "model"
EXPERIMENT_NAME = "ticket-classification"
# Validation accuracy above this bar auto-promotes the model version to
# "Production" in the registry; below it, the version lands in "Staging"
# for manual review. Tune this per business risk tolerance.
PRODUCTION_ACCURACY_THRESHOLD = 0.85


class TicketTorchDataset(torch.utils.data.Dataset):
    """Minimal torch Dataset — tokenizes text into the tensors the HF
    Trainer expects. Kept as a plain torch Dataset (rather than pulling
    in the separate `datasets` library) to keep the dependency surface
    small for a project this size."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def run_training(model_name: str, epochs: int, batch_size: int, lr: float,
                  offline: bool, data_path: str | None = None):
    df = load_raw_data(Path(data_path) if data_path else None)
    train_df, val_df = train_val_split(df)

    clf = TicketClassifier.offline_stub() if offline else TicketClassifier.from_pretrained(model_name)

    train_texts = [clean_text(t) for t in train_df["text"]]
    val_texts = [clean_text(t) for t in val_df["text"]]
    train_labels = [LABEL2ID[label] for label in train_df["label"]]
    val_labels = [LABEL2ID[label] for label in val_df["label"]]

    train_ds = TicketTorchDataset(train_texts, train_labels, clf.tokenizer)
    val_ds = TicketTorchDataset(val_texts, val_labels, clf.tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        learning_rate=lr,
        report_to=[],  # we log to MLflow explicitly below for full control
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=clf.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(clf.tokenizer),
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "base_model": model_name if not offline else "offline-stub-distilbert-tiny",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "git_sha": git_sha(),
        })

        trainer.train()
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({
            "val_accuracy": eval_metrics["eval_accuracy"],
            "val_f1_macro": eval_metrics["eval_f1_macro"],
        })

        # Confusion matrix as an artifact — small, human-checkable sanity
        # check that doesn't require opening the full eval set.
        preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
        cm = confusion_matrix(val_labels, preds, labels=list(range(len(LABELS))))
        cm_path = MODEL_DIR / "confusion_matrix.txt"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cm_path, "w") as f:
            f.write("rows=true label, cols=predicted label\n")
            f.write(f"{LABELS}\n{cm}\n")
        mlflow.log_artifact(str(cm_path))

        # Save model + tokenizer locally too (so the FastAPI service and
        # Docker image can load from a plain path without an MLflow
        # server dependency at serve time).
        clf.save(str(MODEL_DIR))

        # Log via MLflow's native `transformers` flavor rather than raw
        # log_artifacts — this is what makes the run's model resolvable
        # by mlflow.register_model() and loadable with
        # mlflow.transformers.load_model() by anyone with registry access.
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": clf.model, "tokenizer": clf.tokenizer},
            name="model",
            task="text-classification",
            pip_requirements=["transformers", "torch"],
        )

        registered = mlflow.register_model(model_info.model_uri, EXPERIMENT_NAME)

        client = mlflow.tracking.MlflowClient()
        accuracy = eval_metrics["eval_accuracy"]
        client.set_model_version_tag(EXPERIMENT_NAME, registered.version, "val_accuracy", f"{accuracy:.4f}")
        client.set_model_version_tag(EXPERIMENT_NAME, registered.version, "val_f1_macro",
                                      f"{eval_metrics['eval_f1_macro']:.4f}")

        # Registry "aliases" are the current MLflow-recommended replacement
        # for the older Staging/Production "stages" concept (stages are
        # deprecated as of MLflow 2.9). "staging" always gets the new
        # version so it's available for review; "production" only moves
        # forward if the new version actually clears the accuracy bar —
        # this is the promotion gate a CI/CD job checks before a rollout.
        client.set_registered_model_alias(EXPERIMENT_NAME, "staging", registered.version)
        if accuracy >= PRODUCTION_ACCURACY_THRESHOLD:
            client.set_registered_model_alias(EXPERIMENT_NAME, "production", registered.version)
            stage = "production"
        else:
            stage = "staging (below production accuracy threshold)"

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Validation accuracy: {accuracy:.4f} | f1_macro: {eval_metrics['eval_f1_macro']:.4f}")
        print(f"Registered as '{EXPERIMENT_NAME}' v{registered.version} -> stage='{stage}'")
        return run.info.run_id, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--offline", action="store_true",
                         help="Use a tiny randomly-initialized model + local tokenizer "
                              "instead of downloading a pretrained checkpoint. Used for "
                              "fast CI smoke tests.")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()
    run_training(args.model_name, args.epochs, args.batch_size, args.lr,
                 args.offline, args.data_path)


if __name__ == "__main__":
    main()
