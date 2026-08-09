"""
Thin wrapper around a Hugging Face transformers sequence-classification
model.

Design note — two loading modes:
  - "pretrained" (default, used for real training/serving): downloads
    the base model + tokenizer from the Hugging Face Hub, e.g.
    distilbert-base-uncased, and fine-tunes on our labeled data. This
    is what runs in the actual training job / EC2 deployment.
  - "offline" (used by unit tests / fast CI smoke checks): builds a
    tiny randomly-initialized model from a config, with a tokenizer
    trained locally on our own corpus. No network call at all.

Keeping both behind the same interface means unit tests exercise the
*real* code path (tokenize -> model -> logits -> label) without every
CI run needing to download a few hundred MB of pretrained weights.
Integration/training runs (which do need real internet access) use
the pretrained path, same as this would work in GitHub Actions or on
an EC2 box with normal internet access.
"""
import os
from dataclasses import dataclass

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data import ID2LABEL, LABEL2ID, LABELS

DEFAULT_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "distilbert-base-uncased")


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict


class TicketClassifier:
    """Loads a tokenizer + sequence-classification model and exposes a
    simple predict() API used by both training/eval code and the
    FastAPI service."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_name: str = DEFAULT_MODEL_NAME):
        """Real path: fine-tune/serve from a Hugging Face Hub checkpoint.
        Requires internet access (available in GitHub Actions CI and on
        the EC2 deployment target)."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        return cls(tokenizer, model)

    @classmethod
    def from_checkpoint(cls, path: str):
        """Loads a previously fine-tuned model saved to disk (or pulled
        from the MLflow Model Registry — see src/train.py)."""
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(tokenizer, model)

    @classmethod
    def offline_stub(cls, corpus: list[str] | None = None):
        """Fast, network-free path used only by unit tests / CI smoke
        tests. Builds a tiny randomly-initialized DistilBERT and a
        WordPiece tokenizer trained on a small in-repo corpus, so the
        full tokenize -> forward-pass -> label pipeline is exercised
        without downloading any pretrained weights."""
        import tempfile

        from tokenizers import BertWordPieceTokenizer
        from transformers import BertTokenizerFast

        corpus = corpus or [
            "billing invoice refund charge subscription payment",
            "technical error crash bug login upload sync",
            "account password email login two factor delete",
            "general feedback roadmap documentation question",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            corpus_path = os.path.join(tmp, "corpus.txt")
            with open(corpus_path, "w") as f:
                f.write("\n".join(corpus))

            wp_tokenizer = BertWordPieceTokenizer(lowercase=True)
            wp_tokenizer.train([corpus_path], vocab_size=200, min_frequency=1)
            wp_tokenizer.save_model(tmp)

            # BertTokenizerFast reads the vocab file the WordPiece trainer just wrote.
            tokenizer = BertTokenizerFast(vocab_file=os.path.join(tmp, "vocab.txt"),
                                           do_lower_case=True)

        config = AutoConfig.for_model(
            "distilbert",
            vocab_size=tokenizer.vocab_size,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            n_layers=2, n_heads=2, dim=64, hidden_dim=128,  # tiny, for speed
        )
        model = AutoModelForSequenceClassification.from_config(config)
        return cls(tokenizer, model)

    @torch.no_grad()
    def predict(self, text: str) -> Prediction:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 padding=True, max_length=128)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs))
        scores = {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return Prediction(
            label=ID2LABEL[pred_id],
            confidence=round(float(probs[pred_id]), 4),
            scores=scores,
        )

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
