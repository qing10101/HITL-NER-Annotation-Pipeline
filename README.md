# NER Data Label Pipeline

A **Cascading Multi-Agent Inline Boundary Tagging & Audit Engine** for building a
large-scale (20,000+) privacy-NER dataset from noisy e-commerce reviews — without
index-drift or over-annotation errors.

See [PRD.md](PRD.md) for the full spec. The design is taken from `Proposed Pipeline.pdf`.

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

**Tagset:** `MINOR_AGE`, `MINOR_EDU`, `GEN_NOUN`, `GEN_PHYS`, `FAM_KIN`
**Error types:** `RAW_TEXT_MUTATION`, `NON_HUMAN_TAGGING`, `UNANCHORED_TAGGING`, `OMITTED_VALID_TAG`, `MISALLOCATED_LABEL`, `INVALID_SPAN_BOUNDARY`, `OUT_OF_SCOPE_TAG`

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
# JSONL input (sample provided)
python run.py --input data/sample_reviews.jsonl

# CSV input with a custom text column, capped at 500 rows
python run.py --input data/reviews.csv --text-field review --limit 500

# Fresh run into a dedicated dir, ignoring resume state
python run.py --input data/reviews.jsonl --out-dir output/run1 --no-resume
```

CLI flags: `--input` (required), `--text-field` (default `text`), `--id-field`
(default `id`), `--format` (`auto`|`jsonl`|`csv`), `--out-dir`, `--limit`,
`--no-resume`. Re-runs skip `row_id`s already present in `run_log.csv`.

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

## Project layout

```
config.py              env-driven config (keys, model IDs, paths)
run.py                 CLI entrypoint
pipeline/
  ingestion.py         Step 1 — streaming + whitespace normalization
  prompts.py           verbatim Stage 1 + Stage 2 prompts
  providers.py         model-agnostic LLM layer (openai/gemini + factory)
  annotator.py         Step 2 — inline tagging (any provider)
  auditor.py           Step 3 — structured audit/judge (any provider)
  parser.py            Step 4A — deterministic regex index parser
  writers.py           Step 4 — CSV sinks (including annotator cache)
  orchestrator.py      decision fork + batch driver
  schemas.py           pydantic data contracts
scripts/
  prepare_dataset.py   stream-sample 10k rows from Amazon Reviews 2023 (HuggingFace)
  sample_from_dataset.py  randomly select N rows from the 10k dataset
tests/test_parser.py   parser correctness (no API calls)
data/sample_reviews.jsonl
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
- Transient API errors retry with exponential backoff (`tenacity`); a hard failure
  on one row routes it to the review queue with `error_type=PIPELINE_ERROR` rather
  than aborting the batch.
- Output store is CSV (per project decision); the schema is swap-compatible with
  the Postgres target described in the doc.
