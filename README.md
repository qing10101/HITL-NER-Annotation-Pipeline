# NER Data Label Pipeline

A **Cascading Multi-Agent Inline Boundary Tagging & Audit Engine** for building a
large-scale (20,000+) privacy-NER dataset from noisy e-commerce reviews — without
index-drift or over-annotation errors.

The design is taken from `Proposed Pipeline.pdf`.
The annotation rules themselves — what to tag, what to exclude, span boundaries, label
selection — are sourced verbatim from `Final Guideline.docx` ("NER Annotation Guidelines:
Implicit Privacy Risks in Reviews") and embedded as a single `GUIDELINE` block in
[prompts.py](pipeline/prompts.py), shared by both the annotator and auditor system prompts
so the two stages judge against the identical rulebook.

## How it works

```
reviews.jsonl/csv
   │  Step 1  stream row-by-row + strip leading/trailing whitespace
   ▼
Stage 1  ANNOTATOR  (gemini:gemini-3.5-flash) → verbatim text + inline XML tags
   ▼
Stage 2  AUDITOR    (openai:gpt-5.5)          → JSON {status, error_type, auditor_reason}
   │
   ├─ PASS → Step 4A deterministic regex parser → char offsets → gold CSVs
   └─ FAIL → Step 4B review_queue.csv (human-in-the-loop)
```

The LLMs **never** emit coordinates. Offsets are computed by a non-AI `re.finditer`
sweep, so PASS rows are mathematically free of off-by-one / sub-token drift. A
`strip_tags(tagged) == raw` invariant check re-routes any mismatch to humans even
if the auditor passed it.

Either stage can use any provider — the models above are config, not hard-wired
(see [providers.py](pipeline/providers.py)). "Cross-family" auditing just means
the auditor uses a different provider than the annotator.

**Tagset:**
| Label | Category | Meaning |
|---|---|---|
| `MINOR_AGE` | Minor Info | Age/developmental indicator of a real human child under 18 |
| `MINOR_EDU` | Minor Info | A specific educational tier exclusive to human minors (not bare "school") |
| `GEN_NOUN` | Gender | Explicit gendered noun for the reviewer or their romantic partner |
| `GEN_PHYS` | Gender | Sex-specific physiological condition/milestone of the reviewer or partner |
| `FAM_KIN` | Family Structure | Kinship term establishing the reviewer's family network |

**Error types (dominant-selection precedence, first match wins when several apply):**
`RAW_TEXT_MUTATION` > `NON_HUMAN_TAGGING` > `UNANCHORED_TAGGING` > `OUT_OF_SCOPE_TAG` >
`MISALLOCATED_LABEL` > `INVALID_SPAN_BOUNDARY` > `OMITTED_VALID_TAG`

## Setup

```powershell
pip install -r requirements-pipeline.txt
copy .env.example .env   # then fill in GEMINI_API_KEY and OPENAI_API_KEY
```

The benchmark subsystem (`benchmark/`, see below) has its own, heavier dependency
set (sentence-transformers/torch) and is installed separately:

```powershell
pip install -r requirements-benchmark.txt
```

Models are chosen per role as `"<provider>:<model>"` specs. Defaults (current
setup — Gemini annotates, GPT-5.5 audits):

```
ANNOTATOR_MODEL=gemini:gemini-3.5-flash
AUDITOR_MODEL=openai:gpt-5.5
```

Override in `.env` or per-run with `--annotator-model` / `--auditor-model`.
Known providers: `openai`, `gemini`. To swap roles back, flip the two values; to
add a new backend, subclass `LLMProvider` in [providers.py](pipeline/providers.py)
and register it — no other code changes.

## Run

`pipeline/__main__.py` is the CLI entrypoint — run it as a module from the repo
root (`python -m pipeline`), not as a standalone script:

```powershell
# JSONL input
python -m pipeline --input data/test_set_180k.jsonl

# CSV input with a custom text column, capped at 500 rows
python -m pipeline --input data/reviews.csv --text-field review --limit 500

# Fresh run into a dedicated dir, ignoring resume state
python -m pipeline --input data/reviews.jsonl --out-dir output/run1 --no-resume
```

CLI flags: `--input` (required), `--text-field` (default `text`), `--id-field`
(default `id`), `--format` (`auto`|`jsonl`|`csv`), `--out-dir`, `--start`,
`--limit`, `--concurrency` (default `8`), `--delay`, `--no-resume`. Re-runs skip
`row_id`s already present in `run_log.csv`.

`--start` / `--limit` select a **positional window** over the input — rows
`[start, start+limit)` by input row number (0-based). Because the window is
positional (not a count of newly-processed rows), rerunning the *same* window
re-scans the same rows and resume skips the ones already done, rather than
sliding forward into new rows.

```powershell
# First 5000 rows
python -m pipeline --input data/test_set_180k.jsonl --limit 5000

# Rerun the SAME first-5000 window later: finishes only the rows not yet done
python -m pipeline --input data/test_set_180k.jsonl --limit 5000

# Next 5000 rows (5000..9999)
python -m pipeline --input data/test_set_180k.jsonl --start 5000 --limit 5000
```

## Input format

JSONL (one object per line) or CSV, with a text field/column. An `id` is optional —
rows without one get a stable `row_NNNNN` id so every output maps back to its
source row number.

```jsonl
{"id": "r0001", "text": "My 3yo loves this. Perfect gift for a wife too!"}
```

## Outputs (`output/`, all CSV, keyed by `row_id`)

| File | Contents |
|---|---|
| `gold_standard.csv` | PASS rows: `row_id, raw_text, tagged_text, num_entities, entities_json` |
| `gold_spans.csv` | exploded spans: `row_id, entity_index, label, text, start, end` |
| `review_queue.csv` | FAIL rows: `row_id, raw_text, tagged_text, error_type, auditor_reason` |
| `run_log.csv` | every row: `row_id, status, error_type, num_entities, note` |
| `annotator_cache.csv` | annotator results saved immediately after Stage 1 — lets auditor-only retries skip the annotator call |

Offsets are 0-based, half-open: `raw_text[start:end] == text`.

## Human review scripts

Utility scripts live in `scripts/`, grouped by phase, to support dataset
preparation, human annotation, and HITL adjudication workflows. None make any
API calls.

### Dataset prep (`scripts/dataset_prep/`)

**`prepare_dataset.py`** — stream-samples the 180k-row test set from Amazon
Reviews 2023 (HuggingFace), with per-category checkpointing and retry.
```bash
python scripts/dataset_prep/prepare_dataset.py
python scripts/dataset_prep/prepare_dataset.py --out data/my_set.jsonl --seed 99
python scripts/dataset_prep/prepare_dataset.py --no-resume   # discard checkpoints, restart
```

**`sample_from_dataset.py`** — randomly selects N rows from a dataset.
```bash
python scripts/dataset_prep/sample_from_dataset.py --n 500
python scripts/dataset_prep/sample_from_dataset.py --n 200 --seed 99 --out data/my_sample.jsonl
```

**`filter_sample_by_test_set.py`** — removes rows whose `id` appears in
`test_set_180k_minor_edu.csv`, preventing test-set leakage when building
annotation batches from a sampled CSV.
```bash
# default paths (data/sample_2000.csv → data/sample_filtered.csv)
python scripts/dataset_prep/filter_sample_by_test_set.py

# custom paths
python scripts/dataset_prep/filter_sample_by_test_set.py \
    --sample data/my_sample.csv \
    --test-set data/test_set_180k_minor_edu.csv \
    --output data/my_sample_filtered.csv
```
Prints the number of test-set IDs loaded, rows removed, and rows kept.

---

### Review / HITL (`scripts/review/`)

**`export_for_review.py`** — converts pipeline data to spreadsheet-ready CSV
with blank annotation columns.

**`source` mode** — blank annotation sheet from any source JSONL:
```bash
python scripts/review/export_for_review.py source data/test_set_180k.jsonl
# filters
python scripts/review/export_for_review.py source data/test_set_180k.jsonl \
    --category Baby_Products --tier rich --limit 200 \
    --out data/baby_sheet.csv
```
Columns: `row_num, id, category, source_tier, text, annotated_text (blank), reviewer_notes (blank)`

**`queue` mode** — HITL correction sheet from a pipeline output directory:
```bash
python scripts/review/export_for_review.py queue output/my_run
# filter to one error type
python scripts/review/export_for_review.py queue output/my_run \
    --error-type OMITTED_VALID_TAG --out output/my_run/omission_fixes.csv
```
Columns: `row_num, row_id, error_type, auditor_reason, raw_text, faulty_annotated_text, corrected_text (blank), reviewer_notes (blank)`

**`merge_human_reviewed.py`** — reads `output/human_reviewed.csv`, parses each
row's `human_annotation` column through the deterministic regex parser to
compute span offsets, then appends those rows to `output/gold_standard.csv`
and writes the combined result to `output/gold_standard_merged.csv`. Rows with
an empty `human_annotation` (not yet adjudicated), rows that fail tag parsing,
and row_ids already present in gold are skipped and counted.
```bash
python scripts/review/merge_human_reviewed.py
# custom paths
python scripts/review/merge_human_reviewed.py \
    --human-reviewed output/human_reviewed.csv \
    --gold-standard output/gold_standard.csv \
    --output output/gold_standard_merged.csv
```

**`diff_human_vs_gold.py`** — joins a human-annotated sample and the merged
gold standard on `row_id`, parses each side's tagged text into spans, and
writes two diff levels: row-level status (`MATCH | DIFF | PARSE_ERROR |
ONLY_IN_HUMAN | ONLY_IN_GOLD`) to `output/diff_human_vs_gold.csv`, and
span-level classification (`BOUNDARY_SHIFT | LABEL_CONFLICT | HUMAN_ONLY |
GOLD_ONLY`) to `output/span_level_diff.csv`, plus a per-label summary printed
to stdout.
```bash
python scripts/review/diff_human_vs_gold.py
# custom paths
python scripts/review/diff_human_vs_gold.py \
    --human-csv output/sample_500_human.csv \
    --gold-csv output/gold_standard_merged.csv
```

**`make_audit_chunks.py`** — splits gold + review-queue rows into N
reviewable audit chunks (paths hardcoded).

---

### Analysis (`scripts/analysis/`)

**`minor_edu_retrieval.py`** — keyword-screens a source JSONL using a tiered
lexicon of positive triggers (grade levels, school tiers, homeschool
variants), anchor terms (kinship/age nouns, enrollment verbs), and exclusion
terms (occupations, higher-ed, fiction markers, historical self-references,
generic suitability phrases). Outputs a ranked CSV of MINOR_EDU candidates.

**Scoring:** `+2` per trigger match · `+1` per anchor match · `−1` per exclusion match

```bash
# full scan, sorted by confidence
python scripts/analysis/minor_edu_retrieval.py data/test_set_180k.jsonl

# high-confidence rows only (drops product-suitability noise)
python scripts/analysis/minor_edu_retrieval.py data/test_set_180k.jsonl --min-score 3

# focus on a single tier
python scripts/analysis/minor_edu_retrieval.py data/test_set_180k.jsonl \
    --edu-tier high_school --min-score 1

# combined tier sheet
python scripts/analysis/minor_edu_retrieval.py data/test_set_180k.jsonl \
    --edu-tier elementary --edu-tier middle \
    --out data/elem_mid_candidates.csv
```

Output columns: `row_num, row_id, category, source_tier, net_score, edu_tiers, triggers, anchors, exclusions, text, annotated_text (blank), reviewer_notes (blank)`

Tiers: `early_childhood`, `elementary`, `middle`, `high_school`, `homeschool`.
High-score rows (≥ 3) are near-certain genuine child-education mentions;
low/negative scores are typically teacher reviews, product-suitability language,
or historical self-references and can be dropped with `--min-score`.

**`count_entities_by_type.py`** — reads `entities_json` from a gold CSV and
tallies entity spans per label.
```bash
python scripts/analysis/count_entities_by_type.py
python scripts/analysis/count_entities_by_type.py output/gold_standard_merged.csv
```

---

## Benchmark: retriever-equipped local-model comparison (`benchmark/`)

A separate, standalone subsystem — not part of the `pipeline/` labeling path —
for benchmarking a local Ollama generation model against the gold-standard
dataset, GPT-NER-style: retrieve the k most similar (`raw_text -> tagged_text`)
pairs from an embedded datastore as few-shot demonstrations, then prompt a
local model to tag a query sentence.

Embeddings run locally via sentence-transformers using a SimCSE checkpoint
(`princeton-nlp/sup-simcse-bert-base-uncased` by default); Ollama is used only
for the generation call. Install `requirements-benchmark.txt` first (see Setup).

```bash
# 1. Build the embedding datastore once from a gold-standard CSV. When evaluating
#    against a held-out validation set (e.g. validation/general), build the
#    datastore from a CSV that excludes those rows, or the retriever can hand the
#    model near-identical rows back as its own "demonstrations" (retrieval leakage).
#    See output/gold_standard_merged_excl_general500.csv for the general-holdout case.
python benchmark/build_datastore.py \
    --csv output/gold_standard_merged_excl_general500.csv \
    --out-dir benchmark/datastore

# 2. Annotate the same held-out sentences under both conditions, same LLM:

# Retriever-equipped: retrieves k demonstrations per query
python benchmark/annotate.py \
    --datastore-dir benchmark/datastore \
    --gen-model llama3.1:8b \
    --k 8 \
    --input-csv validation/general/gold_standard_merged.csv \
    --out-csv predictions_retriever.csv

# Zero-shot: --k 0 means no demonstrations are retrieved/inserted, same LLM/guideline.
# --compare-with points at the counterpart run's --out-csv: since --input-csv here
# carries gold entities_json, this automatically prints a with-retriever-vs-
# without-retriever P/R/F1 table when the run terminates -- step 3 below is only
# needed to re-run or rescore a comparison after the fact.
python benchmark/annotate.py \
    --datastore-dir benchmark/datastore \
    --gen-model llama3.1:8b \
    --k 0 \
    --input-csv validation/general/gold_standard_merged.csv \
    --out-csv predictions_zeroshot.csv \
    --compare-with predictions_retriever.csv

# 3. (Optional/standalone) score both conditions against gold and compare them side by side
python benchmark/evaluate.py \
    --gold validation/general/gold_standard_merged.csv \
    --pred zero_shot=predictions_zeroshot.csv \
    --pred retriever=predictions_retriever.csv
```

Both `annotate.py` runs are resumable: progress is written to `--out-csv`
incrementally (flushed after every row), so re-running the same command after
an interruption skips rows already done instead of starting over. Pass
`--no-resume` to force a fresh run.

`evaluate.py` reports per-label and micro-averaged precision/recall/F1 (exact
span match: label + start + end offset) for each named `--pred` set, plus a
final side-by-side comparison table — this is what actually establishes
whether the retriever helps, rather than just running `--k 0` in isolation.

By default `annotate.py` uses `ANNOTATOR_SYSTEM_PROMPT` from `pipeline/prompts.py`
as the guideline text (`--guideline-file` overrides it), and both scripts share
`pipeline`'s `TAGSET` and deterministic tag parser (`pipeline/parser.py`) rather
than re-implementing tag parsing — see `benchmark/embeddings.py` for the shared
SimCSE encoder.

## Project layout

```
requirements-pipeline.txt    core pipeline deps (google-genai, openai, pydantic, tenacity, ...)
requirements-benchmark.txt   benchmark-only deps (numpy, pandas, sentence-transformers, ...)
pipeline/
  __main__.py          CLI entrypoint — run as `python -m pipeline`
  config.py            env-driven config (keys, model IDs, paths)
  ingestion.py         Step 1 — streaming + whitespace normalization
  prompts.py           Stage 1 + Stage 2 system/user prompts
  providers.py         model-agnostic LLM layer (openai/gemini + factory)
  annotator.py         Step 2 — inline tagging (any provider)
  auditor.py           Step 3 — structured audit/judge (any provider)
  parser.py            Step 4A — deterministic regex index parser
  writers.py           Step 4 — CSV sinks (including annotator cache)
  orchestrator.py      decision fork + batch driver
  schemas.py           pydantic data contracts
benchmark/
  build_datastore.py   embed a gold CSV into a retrieval datastore (SimCSE)
  annotate.py          retrieve demonstrations + generate via local Ollama model
  embeddings.py         shared SimCSE/sentence-transformers encoder
  evaluate.py           score prediction CSV(s) against gold; compares named conditions (e.g. zero-shot vs retriever)
scripts/
  dataset_prep/
    prepare_dataset.py         stream-sample rows from Amazon Reviews 2023 (HuggingFace)
    sample_from_dataset.py     randomly select N rows from a dataset
    filter_sample_by_test_set.py  remove test-set rows from a sampled CSV
  review/
    export_for_review.py       convert JSONL or review queue to annotation/HITL CSV
    merge_human_reviewed.py    merge adjudicated queue rows into gold_standard
    diff_human_vs_gold.py      row/span-level diff of human annotations vs gold
    make_audit_chunks.py       split gold+queue rows into N reviewable audit chunks (paths hardcoded)
  analysis/
    minor_edu_retrieval.py     keyword-screen a JSONL for MINOR_EDU candidates
    count_entities_by_type.py  tally entity spans per label in a gold CSV
tests/test_parser.py   parser correctness (no API calls)
data/test_set_180k.jsonl
```

## Tests

```powershell
python -m unittest tests.test_parser -v
```

The parser tests run entirely offline (no API keys needed).

## Notes

- Each row costs exactly **2 LLM calls** (1 annotate + 1 audit). Semantic
  disagreements are sent to humans, not retried. On retry after an auditor
  failure, the cached annotator result is reused — only 1 call is charged.
- Rows are processed concurrently (`--concurrency`, default 8). Both API calls
  run in the thread pool via `asyncio.to_thread`, so the event loop is never
  blocked. Use `--concurrency 1` to reproduce sequential behaviour.
- Transient API errors retry with exponential backoff (`tenacity`); a hard failure
  on one row routes it to the review queue with `error_type=PIPELINE_ERROR` rather
  than aborting the batch.
- Output store is CSV (per project decision); the schema is swap-compatible with
  a future Postgres store.
