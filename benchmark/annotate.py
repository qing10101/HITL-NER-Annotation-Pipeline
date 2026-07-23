"""
annotate.py

Steps 2 and 3 of the GPT-NER-style pipeline (sentence-level retrieval variant):
  - embed the query sentence with the same embedding model used to build the datastore
  - kNN search (cosine similarity) over the datastore to pull the k most similar
    retrieved (raw_text -> tagged_text) pairs as few-shot demonstrations
  - build a prompt combining the annotation guideline + retrieved demonstrations + query
  - call a local Ollama generation model to produce the tagged output for the query

This is the "retriever-equipped" condition. To get the "zero-shot" condition for
comparison, just run with --k 0 (no demonstrations retrieved/inserted).

Usage:
    python annotate.py \
        --datastore-dir ./datastore \
        --gen-model llama3.1:8b \
        --embed-model nomic-embed-text \
        --k 8 \
        --input-csv test_reviews.csv \
        --text-col raw_text \
        --out-csv predictions.csv

By default (no --guideline-file), the guideline text is ANNOTATOR_SYSTEM_PROMPT from
pipeline/prompts.py, i.e. the same guideline used by the labeling pipeline itself.
Pass --guideline-file to override it with a different guideline text file.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.prompts import ANNOTATOR_SYSTEM_PROMPT  # noqa: E402

LABELS = ["MINOR_AGE", "MINOR_EDU", "GEN_NOUN", "GEN_PHYS", "FAM_KIN"]

DEFAULT_TASK_INSTRUCTIONS = f"""You are an expert annotator. Tag spans in the input text that match one of
the following entity categories: {', '.join(LABELS)}.

Wrap each tagged span with an inline XML-style tag matching its label, e.g.:
"My <FAM_KIN>great grandson</FAM_KIN> loves this game."

Rules:
- Only tag spans that clearly match one of the categories above.
- Do not tag anything if no entities are present; return the text unchanged.
- Do not alter any text other than inserting the tags.
- Output ONLY the tagged text. No explanation, no preamble, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Ollama calls
# ---------------------------------------------------------------------------

def get_embedding(text: str, model: str, ollama_url: str, max_retries: int = 3) -> np.ndarray:
    url = f"{ollama_url.rstrip('/')}/api/embeddings"
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
            resp.raise_for_status()
            emb = np.array(resp.json()["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Embedding call failed after {max_retries} retries: {last_err}")


def generate(prompt: str, model: str, ollama_url: str, temperature: float = 0.0,
             max_retries: int = 3) -> str:
    """Call Ollama's local generate endpoint. temperature=0.0 for deterministic tagging."""
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Generation call failed after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Retrieval (sentence-level kNN, "Option 4")
# ---------------------------------------------------------------------------

class Datastore:
    def __init__(self, datastore_dir: str):
        ddir = Path(datastore_dir)
        self.embeddings = np.load(ddir / "datastore_embeddings.npy")  # (N, dim), already L2-normalized
        self.meta = pd.read_parquet(ddir / "datastore_meta.parquet")
        assert len(self.meta) == self.embeddings.shape[0], "Embeddings/metadata row count mismatch."

    def top_k(self, query_emb: np.ndarray, k: int) -> pd.DataFrame:
        """Cosine similarity via dot product (both sides pre-normalized)."""
        if k <= 0:
            return self.meta.iloc[0:0]
        sims = self.embeddings @ query_emb  # (N,)
        top_idx = np.argsort(-sims)[:k]
        result = self.meta.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(guideline_text: str, demos: pd.DataFrame, query_text: str) -> str:
    parts = [guideline_text.strip(), "", DEFAULT_TASK_INSTRUCTIONS.strip(), ""]

    if len(demos) > 0:
        parts.append("Examples:")
        for _, row in demos.iterrows():
            parts.append(f"Input: {row['raw_text']}")
            parts.append(f"Output: {row['tagged_text']}")
            parts.append("")

    parts.append("Now tag the following input.")
    parts.append(f"Input: {query_text}")
    parts.append("Output:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output parsing / validation
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(
    r"<(" + "|".join(LABELS) + r")>(.*?)</\1>", re.DOTALL
)


def extract_entities(tagged_text: str) -> list[dict]:
    """Recover an entities_json-style list of {label, text, start, end} from generated tagged text.

    Offsets are computed against the de-tagged (raw) reconstruction of the string,
    so they're comparable to the entities_json convention in the source CSV.
    """
    entities = []
    cursor = 0
    raw_reconstruction = []
    pos = 0
    for m in TAG_PATTERN.finditer(tagged_text):
        # text before this tag, unchanged
        raw_reconstruction.append(tagged_text[pos:m.start()])
        cursor += len(tagged_text[pos:m.start()])
        label, span_text = m.group(1), m.group(2)
        start = cursor
        end = start + len(span_text)
        entities.append({"label": label, "text": span_text, "start": start, "end": end})
        raw_reconstruction.append(span_text)
        cursor = end
        pos = m.end()
    raw_reconstruction.append(tagged_text[pos:])
    return entities


def strip_malformed_tags(tagged_text: str) -> str:
    """Best-effort cleanup if the model emits a tag not in LABELS or an unclosed tag.
    Leaves well-formed tags alone; strips anything that doesn't match TAG_PATTERN's
    known labels by removing stray angle-bracket fragments.
    """
    known_open = {f"<{lab}>" for lab in LABELS} | {f"</{lab}>" for lab in LABELS}
    # remove any tag-like token not in the known set
    def _clean(match):
        tok = match.group(0)
        return tok if tok in known_open else ""
    return re.sub(r"</?[A-Z_]+>", _clean, tagged_text)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retriever-equipped (or zero-shot, with --k 0) LLM NER annotation via Ollama.")
    parser.add_argument("--datastore-dir", required=True)
    parser.add_argument("--guideline-file", default=None,
                         help="Path to a text file containing the annotation guideline. "
                              "Defaults to ANNOTATOR_SYSTEM_PROMPT from pipeline/prompts.py.")
    parser.add_argument("--gen-model", required=True, help="Ollama generation model, e.g. llama3.1:8b, qwen2.5:7b, etc.")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--k", type=int, default=8, help="Number of retrieved demonstrations. Use 0 for zero-shot.")
    parser.add_argument("--input-csv", required=True, help="CSV of new sentences to annotate.")
    parser.add_argument("--text-col", default="raw_text")
    parser.add_argument("--id-col", default="row_id")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows to process (debugging).")
    args = parser.parse_args()

    guideline_text = (
        Path(args.guideline_file).read_text(encoding="utf-8")
        if args.guideline_file else ANNOTATOR_SYSTEM_PROMPT
    )
    ds = Datastore(args.datastore_dir)

    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit)

    results = []
    for i, row in df.iterrows():
        query_text = str(row[args.text_col])
        query_id = row[args.id_col]

        query_emb = get_embedding(query_text, args.embed_model, args.ollama_url)
        demos = ds.top_k(query_emb, args.k)

        prompt = build_prompt(guideline_text, demos, query_text)
        raw_output = generate(prompt, args.gen_model, args.ollama_url)
        cleaned_output = strip_malformed_tags(raw_output)
        entities = extract_entities(cleaned_output)

        results.append({
            args.id_col: query_id,
            "raw_text": query_text,
            "predicted_tagged_text": cleaned_output,
            "predicted_entities_json": json.dumps(entities, ensure_ascii=False),
            "num_predicted_entities": len(entities),
            "retrieved_ids": ",".join(demos["row_id"].astype(str).tolist()) if len(demos) else "",
            "k_used": args.k,
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"  annotated {i + 1}/{len(df)}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions for {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()