"""
hf/annotate.py

Hugging Face (transformers) generation backend for the GPT-NER-style benchmark --
the local-model counterpart to ollama/annotate.py.

Identical pipeline (Steps 2 and 3, sentence-level retrieval variant): embed query
sentences with SimCSE, kNN-retrieve few-shot demonstrations from the datastore,
build the guideline + demos + query prompt, and generate tagged output. The only
difference from the Ollama backend is that generation runs locally through
transformers (weights downloaded from the Hugging Face Hub) instead of via a local
Ollama server. Embeddings still run locally via sentence-transformers/SimCSE
(../embeddings.py), and all retrieval / prompting / parsing / resume / scoring logic
is shared from ../common.py.

The output CSV schema is identical to the Ollama backend's, so evaluate.py and
--compare-with can score runs from either backend interchangeably. LABEL SCHEME and
resume/comparison behaviour are documented in ../common.py.

Generation CLI flags (vs the Ollama backend):
  --gen-model      a Hugging Face model id (e.g. Qwen/Qwen2.5-7B-Instruct,
                   meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it)
  --gen-device     torch device for generation (separate from the embedder's --device)
  --dtype          weight dtype (float16/bfloat16/float32); auto by default
  --load-in-4bit / --load-in-8bit   bitsandbytes quantization (needs a CUDA GPU)
  --max-new-tokens replaces --num-ctx: cap on generated tokens per row
(there is no --ollama-url; nothing is served over HTTP.)

Usage:
    python hf/annotate.py \
        --datastore-dir ./datastore \
        --gen-model Qwen/Qwen2.5-7B-Instruct \
        --embed-model princeton-nlp/sup-simcse-bert-base-uncased \
        --k 8 \
        --input-csv test_reviews.csv \
        --text-col raw_text \
        --out-csv predictions.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# The shared benchmark modules (embeddings, evaluate, common) live one level up in
# benchmark/llm/. This file runs from benchmark/llm/hf/, so put the parent on the
# path before importing them. hf_generation is a sibling in this same folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embeddings import DEFAULT_SIMCSE_MODEL, embed, load_encoder  # noqa: E402
from evaluate import FAILED_MARKER  # noqa: E402
from common import (DEFAULT_GUIDELINE_FILE, Datastore, build_prompt,  # noqa: E402
                    drop_failed_rows, extract_entities, load_progress,
                    report_comparison, strip_malformed_tags)

from hf_generation import HFGenerator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Retriever-equipped (or zero-shot, with --k 0) LLM NER annotation "
                    "via a local Hugging Face transformers model.")
    parser.add_argument("--datastore-dir", required=True)
    parser.add_argument("--guideline-file", default=DEFAULT_GUIDELINE_FILE,
                         help="Path to a text file containing the annotation guideline. Defaults to the "
                              "minimal 3-dimension guideline (benchmark/guidelines/guideline_naive_3dim_"
                              "minimal.txt), whose labels are MINOR, GENDER, KINSHIP.")
    parser.add_argument("--gen-model", required=True,
                         help="Hugging Face model id for the generation model, e.g. "
                              "Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct, "
                              "google/gemma-2-9b-it. Downloaded from the Hub and cached locally.")
    parser.add_argument("--embed-model", default=DEFAULT_SIMCSE_MODEL,
                         help="Hugging Face model id for a SimCSE checkpoint. Must match the model used "
                              "in build_datastore.py, or retrieval similarity scores are meaningless.")
    parser.add_argument("--device", default=None,
                         help="torch device for the embedder (cpu, cuda, mps). Auto-detected if omitted.")
    parser.add_argument("--gen-device", default=None,
                         help="torch device for the generation model (cpu, cuda, mps). Auto-detected if "
                              "omitted. Kept separate from --device so the embedder and generator can sit "
                              "on different devices; ignored under --load-in-4bit/8bit (accelerate places "
                              "the weights).")
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"],
                         help="Weight dtype for the generation model. Auto by default (bfloat16 on "
                              "GPU/MPS, float32 on CPU).")
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="Load the generation model in 4-bit via bitsandbytes (needs a CUDA GPU + "
                              "bitsandbytes + accelerate). Big memory saving for large models.")
    parser.add_argument("--load-in-8bit", action="store_true",
                         help="Load the generation model in 8-bit via bitsandbytes (needs a CUDA GPU + "
                              "bitsandbytes + accelerate).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding query texts.")
    parser.add_argument("--k", type=int, default=8, help="Number of retrieved demonstrations. Use 0 for zero-shot.")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                         help="Cap on tokens generated per row. Tagged output is roughly the input length "
                              "plus a few tokens per tag, so this only needs to comfortably exceed the "
                              "longest input's token count.")
    parser.add_argument("--temperature", type=float, default=0.0,
                         help="Sampling temperature. 0.0 (default) selects greedy decoding for "
                              "deterministic, reproducible tagging.")
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

    # Download + load the generation model once, then reuse for every row.
    print(f"Loading generation model '{args.gen_model}' from the Hugging Face Hub ...")
    generator = HFGenerator(
        args.gen_model, device=args.gen_device, dtype=args.dtype,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(f"Generation model ready on device: {generator.device}")

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
                raw_output = generator.generate(prompt, max_retries=args.max_retries)
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
