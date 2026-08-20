"""
ollama/annotate.py

Steps 2 and 3 of the GPT-NER-style pipeline (sentence-level retrieval variant),
Ollama generation backend:
  - embed the query sentences with the same SimCSE model used to build the datastore
  - kNN search (cosine similarity) over the datastore to pull the k most similar
    retrieved (raw_text -> tagged_text) pairs as few-shot demonstrations
  - build a prompt combining the annotation guideline + retrieved demonstrations + query
  - call a local Ollama generation model to produce the tagged output for the query

Embeddings run locally via sentence-transformers/SimCSE (../embeddings.py); Ollama is
used only for the generation call. The retrieval, prompting, parsing, resume and
scoring logic is backend-agnostic and lives in ../common.py; this file adds only the
Ollama generate() call and the CLI. The Hugging Face backend (../hf/annotate.py)
mirrors it and produces an identical output CSV, so evaluate.py and --compare-with
score runs from either backend interchangeably.

This is the "retriever-equipped" condition. To get the "zero-shot" condition for
comparison, just run with --k 0 (no demonstrations retrieved/inserted).

LABEL SCHEME and resume/comparison behaviour are documented in ../common.py.

Progress is saved incrementally: each row is written to --out-csv (and flushed)
as soon as it's annotated, rather than buffered until the end. If --out-csv
already exists, re-running the same command resumes -- rows whose id already
appears in it are skipped and new rows are appended. Pass --no-resume to
ignore any existing --out-csv and start fresh instead.

Each generation call is retried --max-retries times (default 3, with backoff), each
attempt bounded by --request-timeout seconds. If all retries fail, the row is NOT
allowed to abort the run: it is recorded with a FAILED marker in --out-csv and skipped,
and (being present in the file) is not retried on a later resume. Failed rows are scored
as recall misses -- see evaluate.py. Note a timeout is deterministic given the same
prompt, so all --max-retries attempts will time out identically: raising
--request-timeout (or shortening the prompt) is the fix, not more retries.

If --input-csv carries a gold entities_json column, a "with retriever vs without
retriever" comparison table is printed when the run terminates (pass --compare-with
<the other condition's --out-csv> to score both side by side).

Usage:
    python ollama/annotate.py \
        --datastore-dir ./datastore \
        --gen-model gemma4:e4b \
        --embed-model princeton-nlp/sup-simcse-bert-base-uncased \
        --k 8 \
        --input-csv test_reviews.csv \
        --text-col raw_text \
        --out-csv predictions.csv

By default the guideline text is the minimal 3-dimension prompt
(benchmark/guidelines/guideline_naive_3dim_minimal.txt), which carries no worked
examples -- so the --k 0 condition has no demonstrations at all (a true zero-shot
baseline) and the retriever's demonstrations are the only thing added at --k > 0.
Pass --guideline-file to override it with a different guideline text file (keep its
labels consistent with the 3-dimension scheme, or the parser/scoring won't match).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# The shared benchmark modules (embeddings, evaluate, common) live one level up in
# benchmark/llm/. This file runs from benchmark/llm/ollama/, so put the parent on the
# path before importing them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embeddings import DEFAULT_SIMCSE_MODEL, embed, load_encoder  # noqa: E402
from evaluate import FAILED_MARKER  # noqa: E402
from common import (DEFAULT_GUIDELINE_FILE, Datastore, build_prompt,  # noqa: E402
                    drop_failed_rows, extract_entities, load_progress,
                    report_comparison, strip_malformed_tags)


# ---------------------------------------------------------------------------
# Ollama calls (generation only — embeddings run locally, see ../embeddings.py)
# ---------------------------------------------------------------------------

def generate(prompt: str, model: str, ollama_url: str, temperature: float = 0.0,
             num_ctx: int = 32768, max_retries: int = 3, timeout: int = 180) -> str:
    """Call Ollama's local generate endpoint. temperature=0.0 for deterministic tagging.

    num_ctx must be passed explicitly: Ollama's runtime default context window is 4096
    tokens regardless of what the model supports, and a guideline + demos + query prompt
    routinely exceeds that, so omitting it silently truncates the prompt.

    timeout is the per-attempt HTTP read timeout in seconds. Nothing caps the model's
    output length (no num_predict is sent), so a model that never emits a stop token
    generates until num_ctx fills -- which on a long guideline at a large --num-ctx can
    exceed the default 180s. Such a row exhausts its retries and is recorded FAILED.
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
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Generation call failed after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retriever-equipped (or zero-shot, with --k 0) LLM NER annotation via Ollama.")
    parser.add_argument("--datastore-dir", required=True)
    parser.add_argument("--guideline-file", default=DEFAULT_GUIDELINE_FILE,
                         help="Path to a text file containing the annotation guideline. Defaults to the "
                              "minimal 3-dimension guideline (benchmark/guidelines/guideline_naive_3dim_"
                              "minimal.txt), whose labels are MINOR, GENDER, KINSHIP.")
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
    parser.add_argument("--request-timeout", type=int, default=180,
                         help="Per-attempt HTTP read timeout (seconds) for the Ollama generate call "
                              "(default 180). Nothing caps output length, so a model that fails to stop "
                              "generates until --num-ctx fills; with a long guideline and a large "
                              "--num-ctx that can exceed 180s and the row is recorded FAILED after "
                              "--max-retries attempts. Raise this if FAILED rows show a read timeout.")
    parser.add_argument("--input-csv", required=True, help="CSV of new sentences to annotate.")
    parser.add_argument("--text-col", default="raw_text")
    parser.add_argument("--id-col", default="row_id")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows to process (debugging).")
    parser.add_argument("--max-retries", type=int, default=3,
                         help="Attempts per generation call before giving up on a row (default 3, with "
                              "backoff). A row that fails all attempts is recorded FAILED and skipped, "
                              "not fatal to the run.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing --out-csv and start fresh instead of resuming/appending.")
    parser.add_argument("--retry-failed", action="store_true",
                         help="On resume, re-annotate rows previously recorded FAILED (drops their stale "
                              "FAILED entries from --out-csv first). Default: FAILED rows stay skipped.")
    parser.add_argument("--compare-with", default=None,
                         help="Path to the counterpart condition's --out-csv (e.g. the zero-shot run's "
                              "output, when this run is retriever-equipped, or vice versa). If given and "
                              "--input-csv has a gold entities_json column, prints a with-retriever-vs-"
                              "without-retriever P/R/F1 comparison table when this run terminates.")
    args = parser.parse_args()

    guideline_text = Path(args.guideline_file).read_text(encoding="utf-8")
    ds = Datastore(args.datastore_dir)

    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out_csv)
    fieldnames = [args.id_col, "raw_text", "predicted_tagged_text", "predicted_entities_json",
                  "num_predicted_entities", "retrieved_ids", "k_used"]

    done_ids, failed_ids = (load_progress(out_path, args.id_col)
                            if not args.no_resume else (set(), set()))

    if args.retry_failed and failed_ids:
        dropped = drop_failed_rows(out_path)
        done_ids -= failed_ids  # no longer in the file -> eligible for re-annotation
        print(f"Retrying {dropped} previously-FAILED row(s): removed their stale entries "
              f"from {out_path}, will re-annotate.")

    if done_ids:
        note = f" ({len(failed_ids)} FAILED, left as-is)" if failed_ids and not args.retry_failed else ""
        print(f"Resuming: {len(done_ids)} row(s) already in {out_path}, skipping{note}.")

    todo_df = df[~df[args.id_col].astype(str).isin(done_ids)].reset_index(drop=True)
    if todo_df.empty:
        print(f"Nothing to do -- all {len(df)} row(s) already present in {out_path}.")
        report_comparison(args, df, out_path)
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

        failed = 0
        progress = tqdm(todo_df.iterrows(), total=len(df), initial=len(done_ids),
                         desc=f"annotating (k={args.k})", unit="row")
        for i, row in progress:
            query_text = query_texts[i]
            query_id = row[args.id_col]

            demos = ds.top_k(query_embs[i], args.k)
            prompt = build_prompt(guideline_text, demos, query_text)
            retrieved_ids = ",".join(demos["row_id"].astype(str).tolist()) if len(demos) else ""

            try:
                raw_output = generate(prompt, args.gen_model, args.ollama_url,
                                      num_ctx=args.num_ctx, max_retries=args.max_retries,
                                      timeout=args.request_timeout)
            except RuntimeError as exc:
                # All retries exhausted: record the row as FAILED and skip it rather than
                # aborting the whole run. It scores as a recall miss in evaluate.py.
                failed += 1
                writer.writerow({
                    args.id_col: query_id,
                    "raw_text": query_text,
                    "predicted_tagged_text": "",
                    "predicted_entities_json": FAILED_MARKER,
                    "num_predicted_entities": -1,
                    "retrieved_ids": retrieved_ids,
                    "k_used": args.k,
                })
                f.flush()
                tqdm.write(f"[FAILED] {query_id}: generation failed after "
                           f"{args.max_retries} attempt(s); recorded FAILED and skipped ({exc})")
                progress.set_postfix(failed=failed)
                continue

            cleaned_output = strip_malformed_tags(raw_output)
            entities = extract_entities(cleaned_output)

            writer.writerow({
                args.id_col: query_id,
                "raw_text": query_text,
                "predicted_tagged_text": cleaned_output,
                "predicted_entities_json": json.dumps(entities, ensure_ascii=False),
                "num_predicted_entities": len(entities),
                "retrieved_ids": retrieved_ids,
                "k_used": args.k,
            })
            f.flush()
            progress.set_postfix(entities=len(entities), failed=failed)

    total_done = len(done_ids) + len(todo_df)
    msg = (f"Wrote predictions for {total_done} row(s) total to {args.out_csv} "
           f"({len(todo_df)} newly annotated this run")
    msg += f", {failed} FAILED after retries)." if failed else ")."
    print(msg)

    report_comparison(args, df, out_path)


if __name__ == "__main__":
    main()
