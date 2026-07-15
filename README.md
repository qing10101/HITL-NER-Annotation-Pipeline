# NER Data Label Pipeline

A **Cascading Multi-Agent Inline Boundary Tagging & Audit Engine** for building a
large-scale (20,000+) privacy-NER dataset from noisy e-commerce reviews — without
index-drift or over-annotation errors.

See [PRD.md](PRD.md) for the full spec. The design is taken from `Proposed Pipeline.pdf`.
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
pip install -r requirements.txt
copy .env.example .env   # then fill in GEMINI_API_KEY and OPENAI_API_KEY
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

```powershell
# JSONL input
python run.py --input data/test_set_180k.jsonl

# CSV input with a custom text column, capped at 500 rows
python run.py --input data/reviews.csv --text-field review --limit 500

# Fresh run into a dedicated dir, ignoring resume state
python run.py --input data/reviews.jsonl --out-dir output/run1 --no-resume
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
python run.py --input data/test_set_180k.jsonl --limit 5000

# Rerun the SAME first-5000 window later: finishes only the rows not yet done
python run.py --input data/test_set_180k.jsonl --limit 5000

# Next 5000 rows (5000..9999)
python run.py --input data/test_set_180k.jsonl --start 5000 --limit 5000
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

Two utility scripts live in `scripts/` to support human annotation and HITL
adjudication workflows. Neither makes any API calls.

### `scripts/export_for_review.py` — JSONL / queue → annotation CSV

Converts pipeline data to spreadsheet-ready CSV with blank annotation columns.

**`source` mode** — blank annotation sheet from any source JSONL:
```bash
python scripts/export_for_review.py source data/test_set_180k.jsonl
# filters
python scripts/export_for_review.py source data/test_set_180k.jsonl \
    --category Baby_Products --tier rich --limit 200 \
    --out data/baby_sheet.csv
```
Columns: `row_num, id, category, source_tier, text, annotated_text (blank), reviewer_notes (blank)`

**`queue` mode** — HITL correction sheet from a pipeline output directory:
```bash
python scripts/export_for_review.py queue output/my_run
# filter to one error type
python scripts/export_for_review.py queue output/my_run \
    --error-type OMITTED_VALID_TAG --out output/my_run/omission_fixes.csv
```
Columns: `row_num, row_id, error_type, auditor_reason, raw_text, faulty_annotated_text, corrected_text (blank), reviewer_notes (blank)`

---

### `scripts/minor_edu_retrieval.py` — keyword screening for MINOR_EDU candidates

Scans a source JSONL using a tiered lexicon of positive triggers (grade levels,
school tiers, homeschool variants), anchor terms (kinship/age nouns, enrollment
verbs), and exclusion terms (occupations, higher-ed, fiction markers, historical
self-references, generic suitability phrases). Outputs a ranked CSV of candidates.

**Scoring:** `+2` per trigger match · `+1` per anchor match · `−1` per exclusion match

```bash
# full scan, sorted by confidence
python scripts/minor_edu_retrieval.py data/test_set_180k.jsonl

# high-confidence rows only (drops product-suitability noise)
python scripts/minor_edu_retrieval.py data/test_set_180k.jsonl --min-score 3

# focus on a single tier
python scripts/minor_edu_retrieval.py data/test_set_180k.jsonl \
    --edu-tier high_school --min-score 1

# combined tier sheet
python scripts/minor_edu_retrieval.py data/test_set_180k.jsonl \
    --edu-tier elementary --edu-tier middle \
    --out data/elem_mid_candidates.csv
```

Output columns: `row_num, row_id, category, source_tier, net_score, edu_tiers, triggers, anchors, exclusions, text, annotated_text (blank), reviewer_notes (blank)`

Tiers: `early_childhood`, `elementary`, `middle`, `high_school`, `homeschool`.
High-score rows (≥ 3) are near-certain genuine child-education mentions;
low/negative scores are typically teacher reviews, product-suitability language,
or historical self-references and can be dropped with `--min-score`.

---

### `scripts/filter_sample_by_test_set.py` — remove test-set rows from a sample CSV

Filters out any rows whose `id` appears in `test_set_180k_minor_edu.csv`, preventing
test-set leakage when building annotation batches from a sampled CSV.

```bash
# default paths (data/sample_2000.csv → data/sample_filtered.csv)
python scripts/filter_sample_by_test_set.py

# custom paths
python scripts/filter_sample_by_test_set.py \
    --sample data/my_sample.csv \
    --test-set data/test_set_180k_minor_edu.csv \
    --output data/my_sample_filtered.csv
```

Prints the number of test-set IDs loaded, rows removed, and rows kept.

---

## Project layout

```
config.py              env-driven config (keys, model IDs, paths)
run.py                 CLI entrypoint
pipeline/
  ingestion.py         Step 1 — streaming + whitespace normalization
  prompts.py           Stage 1 + Stage 2 system/user prompts
  providers.py         model-agnostic LLM layer (openai/gemini + factory)
  annotator.py         Step 2 — inline tagging (any provider)
  auditor.py           Step 3 — structured audit/judge (any provider)
  parser.py            Step 4A — deterministic regex index parser
  writers.py           Step 4 — CSV sinks (including annotator cache)
  orchestrator.py      decision fork + batch driver
  schemas.py           pydantic data contracts
scripts/
  prepare_dataset.py      stream-sample rows from Amazon Reviews 2023 (HuggingFace)
  sample_from_dataset.py  randomly select N rows from a dataset
  export_for_review.py         convert JSONL or review queue to annotation/HITL CSV
  minor_edu_retrieval.py       keyword-screen a JSONL for MINOR_EDU candidates
  filter_sample_by_test_set.py remove test-set rows from a sampled CSV
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
  the Postgres target described in the doc.
