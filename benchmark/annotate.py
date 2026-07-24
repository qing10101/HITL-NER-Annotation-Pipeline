"""
annotate.py

Steps 2 and 3 of the GPT-NER-style pipeline (sentence-level retrieval variant):
  - embed the query sentences with the same SimCSE model used to build the datastore
  - kNN search (cosine similarity) over the datastore to pull the k most similar
    retrieved (raw_text -> tagged_text) pairs as few-shot demonstrations
  - build a prompt combining the annotation guideline + retrieved demonstrations + query
  - call a local Ollama generation model to produce the tagged output for the query

Embeddings run locally via sentence-transformers/SimCSE (embeddings.py); Ollama is used
only for the generation call.

This is the "retriever-equipped" condition. To get the "zero-shot" condition for
comparison, just run with --k 0 (no demonstrations retrieved/inserted).

Progress is saved incrementally: each row is written to --out-csv (and flushed)
as soon as it's annotated, rather than buffered until the end. If --out-csv
already exists, re-running the same command resumes -- rows whose id already
appears in it are skipped and new rows are appended. Pass --no-resume to
ignore any existing --out-csv and start fresh instead.

Usage:
    python annotate.py \
        --datastore-dir ./datastore \
        --gen-model gemma4:e4b \
        --embed-model princeton-nlp/sup-simcse-bert-base-uncased \
        --k 8 \
        --input-csv test_reviews.csv \
        --text-col raw_text \
        --out-csv predictions.csv

By default (no --guideline-file), the guideline text is ANNOTATOR_SYSTEM_PROMPT from
pipeline/prompts.py, i.e. the same guideline used by the labeling pipeline itself.
Pass --guideline-file to override it with a different guideline text file.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from embeddings import DEFAULT_SIMCSE_MODEL, embed, load_encoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import TAGSET  # noqa: E402
from pipeline.parser import TagParseError, parse_tagged_text  # noqa: E402
from pipeline.prompts import ANNOTATOR_SYSTEM_PROMPT  # noqa: E402

LABELS = list(TAGSET)

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
# Ollama calls (generation only — embeddings run locally, see embeddings.py)
# ---------------------------------------------------------------------------

def generate(prompt: str, model: str, ollama_url: str, temperature: float = 0.0,
             num_ctx: int = 32768, max_retries: int = 3) -> str:
    """Call Ollama's local generate endpoint. temperature=0.0 for deterministic tagging.

    num_ctx must be passed explicitly: Ollama's runtime default context window is 4096
    tokens regardless of what the model supports, and a guideline + demos + query prompt
    routinely exceeds that, so omitting it silently truncates the prompt.
    """
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
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

def extract_entities(tagged_text: str) -> list[dict]:
    """Recover an entities_json-style list of {label, text, start, end} from generated tagged text.

    Delegates to the pipeline's deterministic tag parser (pipeline/parser.py) so the
    benchmark uses the exact same offset-counting logic as the labeling pipeline itself.
    Unlike pipeline callers, benchmark generations come from an unaudited local model and
    aren't guaranteed well-formed, so malformed tag structure (unbalanced/mismatched) is
    swallowed into zero entities rather than raised.
    """
    try:
        _, spans = parse_tagged_text(tagged_text)
    except TagParseError:
        return []
    return [{"label": s.label, "text": s.text, "start": s.start, "end": s.end} for s in spans]


def strip_malformed_tags(tagged_text: str) -> str:
    """Best-effort cleanup if the model emits a tag not in LABELS or an unclosed tag.
    Leaves well-formed tags alone; strips anything that doesn't match a known
    label by removing stray angle-bracket fragments.
    """
    known_open = {f"<{lab}>" for lab in LABELS} | {f"</{lab}>" for lab in LABELS}
    # remove any tag-like token not in the known set
    def _clean(match):
        tok = match.group(0)
        return tok if tok in known_open else ""
    return re.sub(r"</?[A-Z_]+>", _clean, tagged_text)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_ids(out_path: Path, id_col: str) -> set[str]:
    """Row ids already written to a prior (possibly interrupted) --out-csv run.

    Tolerates a missing, empty, or header-only file (all -> no ids done yet)
    so a half-written file from a crash mid-flush doesn't break resume.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or id_col not in reader.fieldnames:
            return set()
        return {row[id_col] for row in reader}


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
    parser.add_argument("--embed-model", default=DEFAULT_SIMCSE_MODEL,
                         help="Hugging Face model id for a SimCSE checkpoint. Must match the model used "
                              "in build_datastore.py, or retrieval similarity scores are meaningless.")
    parser.add_argument("--device", default=None, help="torch device (cpu, cuda, mps). Auto-detected if omitted.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding query texts.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL, used for the generation call.")
    parser.add_argument("--k", type=int, default=8, help="Number of retrieved demonstrations. Use 0 for zero-shot.")
    parser.add_argument("--num-ctx", type=int, default=32768,
                         help="Ollama context window (tokens) for generation. Ollama's runtime default "
                              "is only 4096 regardless of what the model supports, which a guideline + "
                              "demos + query prompt routinely exceeds, so this must be set explicitly.")
    parser.add_argument("--input-csv", required=True, help="CSV of new sentences to annotate.")
    parser.add_argument("--text-col", default="raw_text")
    parser.add_argument("--id-col", default="row_id")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows to process (debugging).")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing --out-csv and start fresh instead of resuming/appending.")
    args = parser.parse_args()

    guideline_text = (
        Path(args.guideline_file).read_text(encoding="utf-8")
        if args.guideline_file else ANNOTATOR_SYSTEM_PROMPT
    )
    ds = Datastore(args.datastore_dir)

    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out_csv)
    fieldnames = [args.id_col, "raw_text", "predicted_tagged_text", "predicted_entities_json",
                  "num_predicted_entities", "retrieved_ids", "k_used"]

    done_ids = load_done_ids(out_path, args.id_col) if not args.no_resume else set()
    if done_ids:
        print(f"Resuming: {len(done_ids)} row(s) already in {out_path}, skipping.")

    todo_df = df[~df[args.id_col].astype(str).isin(done_ids)].reset_index(drop=True)
    if todo_df.empty:
        print(f"Nothing to do -- all {len(df)} row(s) already present in {out_path}.")
        return

    encoder = load_encoder(args.embed_model, args.device)
    query_texts = todo_df[args.text_col].astype(str).tolist()
    query_embs = embed(encoder, query_texts, batch_size=args.batch_size)

    # Append if resuming into an existing file; otherwise start the file fresh.
    append = bool(done_ids)
    with out_path.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or out_path.stat().st_size == 0:
            writer.writeheader()
            f.flush()

        progress = tqdm(todo_df.iterrows(), total=len(df), initial=len(done_ids),
                         desc=f"annotating (k={args.k})", unit="row")
        for i, row in progress:
            query_text = query_texts[i]
            query_id = row[args.id_col]

            demos = ds.top_k(query_embs[i], args.k)

            prompt = build_prompt(guideline_text, demos, query_text)
            raw_output = generate(prompt, args.gen_model, args.ollama_url, num_ctx=args.num_ctx)
            cleaned_output = strip_malformed_tags(raw_output)
            entities = extract_entities(cleaned_output)

            writer.writerow({
                args.id_col: query_id,
                "raw_text": query_text,
                "predicted_tagged_text": cleaned_output,
                "predicted_entities_json": json.dumps(entities, ensure_ascii=False),
                "num_predicted_entities": len(entities),
                "retrieved_ids": ",".join(demos["row_id"].astype(str).tolist()) if len(demos) else "",
                "k_used": args.k,
            })
            f.flush()
            progress.set_postfix(entities=len(entities))

    total_done = len(done_ids) + len(todo_df)
    print(f"Wrote predictions for {total_done} row(s) total to {args.out_csv} "
          f"({len(todo_df)} newly annotated this run).")


if __name__ == "__main__":
    main()