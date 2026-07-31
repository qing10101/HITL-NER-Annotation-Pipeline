"""train_gliner.py

Fine-tunes GLiNER (urchade/gliner_medium-v2.1) on the shared split.

GLiNER matches candidate text spans against natural-language label phrases
in a shared embedding space, so the training data (built by prepare_data.py)
uses the phrases from common.GLINER_LABEL_PHRASES instead of the raw label
codes — and predict.py must query with those exact phrases.

Input format (data/gliner/{train,dev}.json), per example:
    {"tokenized_text": ["My", "great", "grandson", ...],
     "ner": [[1, 2, "kinship term for the reviewer's family member"]]}
with token indices and an INCLUSIVE end index.

Usage:
    python benchmark/supervised/train_gliner.py --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments

import common  # noqa: F401  (keeps the label-phrase contract in one place)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Fine-tune GLiNER.")
    ap.add_argument("--data-dir", default=str(here / "data"))
    ap.add_argument("--out-dir", default=str(here / "models" / "gliner"))
    ap.add_argument(
        "--checkpoint-dir", default=None,
        help="Where to write per-epoch checkpoints (default: <out-dir>/checkpoints). "
             "Point this at local disk when --out-dir is a network/Drive mount.",
    )
    ap.add_argument("--base-model", default="urchade/gliner_medium-v2.1")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="Learning rate for the pretrained encoder.")
    ap.add_argument("--others-lr", type=float, default=1e-5,
                    help="Learning rate for the non-pretrained heads.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "gliner" / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "gliner" / "dev.json", encoding="utf-8") as f:
        dev_data = json.load(f)
    print(f"train={len(train_data)} dev={len(dev_data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(args.base_model)
    model.to(device)

    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=args.lr,
        weight_decay=0.01,
        others_lr=args.others_lr,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=0,
        use_cpu=(device == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    best_dir = out_dir / "model-best"
    model.save_pretrained(str(best_dir))
    print(f"Done. Best model: {best_dir}")


if __name__ == "__main__":
    main()
