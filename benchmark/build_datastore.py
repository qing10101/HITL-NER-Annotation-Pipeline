"""
build_datastore.py

Step 1 of the GPT-NER-style pipeline (sentence-level retrieval variant, "Option 4").

Reads the 20k silver-annotated CSV (columns: row_id, raw_text, tagged_text,
num_entities, entities_json), embeds every row's raw_text with a SimCSE
sentence-embedding model (via sentence-transformers), and saves:

  - datastore_embeddings.npy   -> float32 array, shape (N, dim), L2-normalized
  - datastore_meta.parquet     -> row_id, raw_text, tagged_text, num_entities, entities_json

These two files together ARE the datastore. Run this once; re-run only if
the underlying 20k corpus or the embedding model changes.

Usage:
    python build_datastore.py \
        --csv silver_20k.csv \
        --out-dir ./datastore \
        --embed-model princeton-nlp/sup-simcse-bert-base-uncased
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from embeddings import DEFAULT_SIMCSE_MODEL, embed, load_encoder


def main():
    parser = argparse.ArgumentParser(description="Build the sentence-level embedding datastore.")
    parser.add_argument("--csv", required=True, help="Path to the 20k silver-annotated CSV.")
    parser.add_argument("--out-dir", required=True, help="Directory to write datastore files to.")
    parser.add_argument("--embed-model", default=DEFAULT_SIMCSE_MODEL,
                         help="Hugging Face model id for a SimCSE checkpoint, e.g. "
                              "princeton-nlp/sup-simcse-bert-base-uncased or "
                              "princeton-nlp/unsup-simcse-roberta-large.")
    parser.add_argument("--device", default=None, help="torch device (cpu, cuda, mps). Auto-detected if omitted.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-col", default="raw_text",
                         help="Which column to embed. Use 'raw_text' (untagged) so the query embedding "
                              "at inference time is comparable — you never have tagged_text for new inputs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = {"row_id", "raw_text", "tagged_text", "num_entities", "entities_json"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {missing}")

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Embedding column '{args.text_col}' with SimCSE model '{args.embed_model}' ...")

    encoder = load_encoder(args.embed_model, args.device)
    texts = df[args.text_col].astype(str).tolist()
    emb_matrix = embed(encoder, texts, batch_size=args.batch_size)

    np.save(out_dir / "datastore_embeddings.npy", emb_matrix)
    df[["row_id", "raw_text", "tagged_text", "num_entities", "entities_json"]].to_parquet(
        out_dir / "datastore_meta.parquet", index=False
    )

    print(f"Saved embeddings: {out_dir / 'datastore_embeddings.npy'}  shape={emb_matrix.shape}")
    print(f"Saved metadata:   {out_dir / 'datastore_meta.parquet'}  rows={len(df)}")
    print("Datastore build complete.")


if __name__ == "__main__":
    main()
