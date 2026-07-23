"""
build_datastore.py

Step 1 of the GPT-NER-style pipeline (sentence-level retrieval variant, "Option 4").

Reads the 20k silver-annotated CSV (columns: row_id, raw_text, tagged_text,
num_entities, entities_json), embeds every row's raw_text using a local
Ollama embedding model, and saves:

  - datastore_embeddings.npy   -> float32 array, shape (N, dim), L2-normalized
  - datastore_meta.parquet     -> row_id, raw_text, tagged_text, num_entities, entities_json

These two files together ARE the datastore. Run this once; re-run only if
the underlying 20k corpus changes.

Usage:
    python build_datastore.py \
        --csv silver_20k.csv \
        --out-dir ./datastore \
        --embed-model nomic-embed-text \
        --ollama-url http://localhost:11434
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def get_embedding(text: str, model: str, ollama_url: str, max_retries: int = 3) -> list[float]:
    """Call Ollama's local embeddings endpoint for a single string."""
    url = f"{ollama_url.rstrip('/')}/api/embeddings"
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["embedding"]
        except Exception as e:  # noqa: BLE001 - we want to retry on any transient error
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Embedding call failed after {max_retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Build the sentence-level embedding datastore.")
    parser.add_argument("--csv", required=True, help="Path to the 20k silver-annotated CSV.")
    parser.add_argument("--out-dir", required=True, help="Directory to write datastore files to.")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                         help="Ollama embedding model name (must be pulled already, e.g. `ollama pull nomic-embed-text`).")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL of the local Ollama server.")
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
    print(f"Embedding column '{args.text_col}' with model '{args.embed_model}' via {args.ollama_url} ...")

    embeddings = []
    for i, text in enumerate(df[args.text_col].astype(str).tolist()):
        emb = get_embedding(text, args.embed_model, args.ollama_url)
        embeddings.append(emb)
        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"  embedded {i + 1}/{len(df)}")

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # L2-normalize so cosine similarity == dot product at retrieval time.
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # guard against a zero vector
    emb_matrix = emb_matrix / norms

    np.save(out_dir / "datastore_embeddings.npy", emb_matrix)
    df[["row_id", "raw_text", "tagged_text", "num_entities", "entities_json"]].to_parquet(
        out_dir / "datastore_meta.parquet", index=False
    )

    print(f"Saved embeddings: {out_dir / 'datastore_embeddings.npy'}  shape={emb_matrix.shape}")
    print(f"Saved metadata:   {out_dir / 'datastore_meta.parquet'}  rows={len(df)}")
    print("Datastore build complete.")


if __name__ == "__main__":
    main()