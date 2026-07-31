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

model-best is the epoch with the highest dev F1 (GLiNER's own exact-match
span scorer), not the lowest dev loss — see F1Trainer.

Usage:
    python benchmark/supervised/train_gliner.py --epochs 3
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments

import common  # noqa: F401  (keeps the label-phrase contract in one place)


def make_collator(model, entity_types: list[str]):
    """Build a DataCollator pinned to a fixed label set.

    By default GLiNER derives each batch's class list from the labels actually
    present in that batch (processor.collate_raw_batch -> if entity_types is
    None: batch_generate_class_mappings). Three quarters of our rows have no
    entities, so ~10% of batches (0.75 ** batch_size) come out with zero
    classes and die on a `[batch, -1, 0]` reshape; GLiNER's Trainer catches
    that and skips the step, silently dropping those gradients.

    Passing entity_types takes the predefined-mapping branch instead, so the
    same three phrases are the candidate set for every batch. Beyond stopping
    the skips this is the better training signal: a row containing only
    KINSHIP otherwise teaches the model nothing about MINOR or GENDER being
    absent from it, because those phrases are never shown as candidates.

    The list must be per-example, not a single flat list: with a flat list the
    processor builds one shared class_to_ids dict, but create_labels indexes it
    as batch['classes_to_id'][i] and dies with KeyError: 0. Batch size varies
    (final batch of an epoch), so it has to be rebuilt per call.

    Written against gliner 0.2.22, which requirements-supervised.txt pins
    exactly. The __call__ branch below is a hedge for the 0.2.28 shape, but
    that release also renames DataCollator to SpanDataCollator, so the import
    above would need updating too — the pin is the real contract here.
    """
    collator = DataCollator(
        model.config, data_processor=model.data_processor, prepare_labels=True,
    )
    call_takes_types = "entity_types" in inspect.signature(collator.__call__).parameters

    def collate(input_x, **call_kwargs):
        per_example = [entity_types] * len(input_x)
        if call_takes_types:
            call_kwargs.setdefault("entity_types", per_example)
            return collator(input_x, **call_kwargs)
        collator.entity_types = per_example
        return collator(input_x, **call_kwargs)

    return collate


class F1Trainer(Trainer):
    """Trainer that scores dev F1 each epoch and snapshots the best model.

    GLiNER's Trainer passes no compute_metrics, so HF's checkpoint selection
    would fall back to its `metric_for_best_model=None` default of eval_loss —
    which routinely peaks an epoch away from F1. Here each eval additionally
    runs GLiNER's own span-level scorer (exact match on label + span offsets,
    micro-averaged) and writes model-best whenever F1 improves.

    We snapshot directly instead of using `load_best_model_at_end`: that path
    reloads a checkpoint with a plain `load_state_dict(..., strict=False)`,
    which can silently no-op against GLiNER's custom save format and leave you
    with the final epoch's weights while reporting success.
    """

    def __init__(self, *args, gliner_model, dev_data, entity_types, best_dir,
                 eval_threshold=0.5, eval_batch_size=12, **kwargs):
        super().__init__(*args, **kwargs)
        self._gliner = gliner_model
        self._dev_data = dev_data
        self._entity_types = entity_types
        self._best_dir = Path(best_dir)
        self._eval_threshold = eval_threshold
        self._eval_batch_size = eval_batch_size
        self.best_f1 = -1.0
        self.best_epoch = None

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        report, f1 = self._gliner.evaluate(
            self._dev_data,
            flat_ner=True,
            threshold=self._eval_threshold,
            batch_size=self._eval_batch_size,
            entity_types=self._entity_types,
        )
        epoch = self.state.epoch or 0.0
        metrics["eval_overall_f1"] = f1
        self.log({"eval_overall_f1": f1})
        print(f"[epoch {epoch:.2f}] dev {report.strip()}")
        if f1 > self.best_f1:
            self.best_f1, self.best_epoch = f1, epoch
            self._gliner.save_pretrained(str(self._best_dir))
            print(f"[epoch {epoch:.2f}] new best F1={f1:.4f} -> {self._best_dir}")
        self._gliner.train()  # model.evaluate() left it in eval mode
        return metrics


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
    ap.add_argument("--eval-threshold", type=float, default=0.5,
                    help="Span-score threshold used for the per-epoch dev F1 that "
                         "selects model-best. Match it to predict.py --threshold.")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "gliner" / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "gliner" / "dev.json", encoding="utf-8") as f:
        dev_data = json.load(f)
    n_empty = sum(1 for ex in train_data if not ex["ner"])
    print(f"train={len(train_data)} ({n_empty} with no entities) dev={len(dev_data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(args.base_model)
    model.to(device)

    phrases = list(common.PHRASE_TO_LABEL)
    data_collator = make_collator(model, phrases)

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
        # F1Trainer snapshots model-best itself; HF only keeps a resume checkpoint.
        load_best_model_at_end=False,
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=0,
        use_cpu=(device == "cpu"),
        report_to="none",
    )

    best_dir = out_dir / "model-best"
    trainer = F1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        gliner_model=model,
        dev_data=dev_data,
        entity_types=phrases,
        best_dir=best_dir,
        eval_threshold=args.eval_threshold,
        eval_batch_size=args.batch_size,
    )
    trainer.train()

    if trainer.best_epoch is None:
        # No eval ever ran (e.g. --epochs < 1); fall back to the final weights.
        model.save_pretrained(str(best_dir))
        print(f"Done (no dev eval ran). Final model: {best_dir}")
    else:
        print(f"Done. Best dev F1={trainer.best_f1:.4f} at epoch "
              f"{trainer.best_epoch:.2f}. Best model: {best_dir}")


if __name__ == "__main__":
    main()
