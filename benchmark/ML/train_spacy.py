"""train_spacy.py

Trains a spaCy NER pipeline on the shared split produced by prepare_data.py.

The gold char spans are attached with doc.char_span(); spans that don't land
on spaCy token boundaries are expanded (alignment_mode="expand") and any
overlaps that expansion creates are resolved with filter_spans — both are
counted and reported rather than silently dropped.

Two variants:
    default : tok2vec + ner (fast CPU/GPU baseline)
    --trf   : roberta-base transformer + ner (needs spacy-transformers)

Usage:
    python benchmark/ML/train_spacy.py                # tok2vec baseline
    python benchmark/ML/train_spacy.py --trf --gpu 0  # transformer variant
"""
from __future__ import annotations

import argparse
from pathlib import Path

import spacy
from spacy.cli.init_config import init_config
from spacy.cli.train import train as spacy_train
from spacy.tokens import DocBin
from spacy.util import filter_spans

import common


def build_docbin(nlp, examples: list[common.Example]) -> tuple[DocBin, int, int]:
    """Convert examples to a DocBin; returns (docbin, n_expanded, n_dropped)."""
    db = DocBin()
    expanded = dropped = 0
    for ex in examples:
        doc = nlp.make_doc(ex.text)
        spans = []
        for ent in ex.entities:
            span = doc.char_span(
                ent["start"], ent["end"], label=ent["label"],
                alignment_mode="expand",
            )
            if span is None:
                dropped += 1
                continue
            if (span.start_char, span.end_char) != (ent["start"], ent["end"]):
                expanded += 1
            spans.append(span)
        kept = filter_spans(spans)
        dropped += len(spans) - len(kept)
        doc.ents = kept
        db.add(doc)
    return db, expanded, dropped


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Train spaCy NER on the shared split.")
    ap.add_argument("--data-dir", default=str(here / "data"))
    ap.add_argument("--out-dir", default=str(here / "models" / "spacy"))
    ap.add_argument("--trf", action="store_true",
                    help="Use a roberta-base transformer encoder instead of tok2vec.")
    ap.add_argument("--gpu", type=int, default=-1,
                    help="GPU id (-1 = CPU). Required in practice for --trf.")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spacy_data = data_dir / "spacy"
    spacy_data.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("en")
    for split in ("train", "dev"):
        examples = common.load_examples(data_dir / "splits" / f"{split}.csv")
        db, expanded, dropped = build_docbin(nlp, examples)
        db.to_disk(spacy_data / f"{split}.spacy")
        print(f"[{split}] {len(examples)} docs -> {split}.spacy "
              f"(spans expanded={expanded}, dropped={dropped})")

    config = init_config(
        lang="en",
        pipeline=["ner"],
        optimize="accuracy" if args.trf else "efficiency",
        gpu=args.trf,
        silent=True,
    )
    config["paths"]["train"] = str(spacy_data / "train.spacy")
    config["paths"]["dev"] = str(spacy_data / "dev.spacy")
    config_path = out_dir / "config.cfg"
    config.to_disk(config_path)
    print(f"Config written to {config_path}")

    spacy_train(config_path, output_path=out_dir, use_gpu=args.gpu)
    print(f"Done. Best model: {out_dir / 'model-best'}")


if __name__ == "__main__":
    main()
