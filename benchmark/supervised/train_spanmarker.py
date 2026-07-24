"""train_spanmarker.py

Fine-tunes SpanMarker with a DeBERTa-v3-base encoder on the shared split.

SpanMarker reframes NER as span classification: it enumerates candidate
spans (up to --entity-max-length words), wraps each in marker tokens, and
classifies the marker representation into a label or "no entity". It
consumes token + BIO-tag datasets and returns char offsets natively at
prediction time.

Defaults are sized for a 12 GB GPU (batch 8 x grad-accum 2, fp16).

Usage:
    python benchmark/ML/train_spanmarker.py --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Sequence, Value
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments

import common


def load_split(path: Path) -> Dataset:
    tokens, ner_tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens.append(row["tokens"])
            ner_tags.append([common.BIO_TAG_TO_ID[t] for t in row["tags"]])
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=common.BIO_TAGS)),
    })
    return Dataset.from_dict({"tokens": tokens, "ner_tags": ner_tags},
                             features=features)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Fine-tune SpanMarker (DeBERTa-v3).")
    ap.add_argument("--data-dir", default=str(here / "data"))
    ap.add_argument("--out-dir", default=str(here / "models" / "spanmarker"))
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--model-max-length", type=int, default=256)
    ap.add_argument("--entity-max-length", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    train_ds = load_split(data_dir / "bio" / "train.jsonl")
    dev_ds = load_split(data_dir / "bio" / "dev.jsonl")
    print(f"train={len(train_ds)} dev={len(dev_ds)}")

    model = SpanMarkerModel.from_pretrained(
        args.encoder,
        labels=common.BIO_TAGS,
        model_max_length=args.model_max_length,
        marker_max_length=128,
        entity_max_length=args.entity_max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_overall_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    best_dir = out_dir / "model-best"
    trainer.save_model(str(best_dir))
    print(f"Done. Best model: {best_dir}")


if __name__ == "__main__":
    main()
