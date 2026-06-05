# PRD — Cascading Multi-Agent Inline Boundary Tagging & Audit Engine

> Privacy-NER data labeling pipeline for large-scale (20,000+) e-commerce review
> annotation, derived from `Proposed Pipeline.pdf`.

- **Status:** Draft → Implementation
- **Owner:** zhirong@aiecd.com
- **Date:** 2026-06-04
- **Models:** Stage 1 annotator = `gemini-3.5-flash`; Stage 2 auditor = `gpt-5.4-mini` (configurable via `.env`, doc values are defaults)

---

## 1. Problem & Goal

We need a high-fidelity labeled dataset of **≥ 20,000 sequences** for privacy NER over
noisy Amazon-style consumer reviews, **without** propagating two failure classes that
plague generative labeling:

1. **Index-drift errors** — off-by-one / sub-token span offsets produced when an LLM is
   asked to emit character coordinates directly.
2. **Data-imbalance / over-annotation anomalies** — tagging pets, fictional personas,
   gift recipients, or the reviewer's own recalled childhood.

**Core idea:** reframe annotation from *coordinate generation* into a *deterministic
text-rewriting + verification cycle*. The LLM never emits offsets; it only injects inline
XML tags. A non-AI regex parser derives exact `start`/`end` offsets afterward. A
**cross-family** second model audits the rewrite to break single-model confirmation bias.

**Success criteria**

| Metric | Target |
|---|---|
| Throughput | Stream + label 20,000+ rows row-by-row without memory exhaustion |
| Index correctness | 0 index-drift errors on PASS rows (offsets are computed, never generated) |
| Character fidelity | `raw_text == strip_tags(tagged_text)` for every committed row |
| Human overhead | Only FAIL rows reach humans; PASS rows are fully automated |
| Auditability | Every FAIL row carries `error_type` + one-sentence `auditor_reason` |

---

## 2. Tagset (5 privacy categories)

| Tag | Meaning | Key inclusions | Key exclusions |
|---|---|---|---|
| `MINOR_AGE` | Age / milestone / age-bracket of a **real, living human child < 18** | "3yo", "toddler", "newborn", "as a minor myself"; minor-denoting kinship/count nouns ("twins", "stepson" in middle school) | pet ages, "my 5-year-old car", reviewer's own past childhood |
| `MINOR_EDU` | Academic tier **exclusive to human minors** | "kindergarten", "5th grade", "high school freshman" | "college", "university", "trade school", pet training |
| `GEN_NOUN` | Gendered identity noun anchoring the **reviewer or their direct partner** | "husband", "wife", "hubby", "fiancée", "as a mom of two" | third-party gendered nouns ("the female cashier"), gift-recipient "wife" |
| `GEN_PHYS` | Gender-specific biological/medical state of reviewer/partner | "breastfeeding", "postpartum", "third trimester", "miscarriage" | non-gender-specific conditions ("chest pain", "hair loss") |
| `FAM_KIN` | Kinship markers in the reviewer's active household | "mother-in-law", "twin sister", "nephew", "my adult son" | kinship noun that resolves to a minor (→ `MINOR_AGE`) |

**Span / compound / priority rules** (enforced in the Stage-1 prompt, audited in Stage 2):

- **Span strategy:** tag the entire noun phrase carrying the risk, excluding trailing punctuation.
- **Demographic compounds:** age + gendered minor noun → one `MINOR_AGE` span ("16-year-old girl"), do not split.
- **Flat-NER priority:** when a token is both kinship and minor and inseparable, prefer `MINOR_AGE`.
- **Pregnancy-loss:** miscarriage is `GEN_PHYS`, never a phantom `MINOR_AGE`.
- **Historical self-reference:** reviewer's own past childhood (past tense) → tag nothing.

---

## 3. Architecture

```
 reviews.jsonl/csv
        │  Step 1: stream row-by-row + strip leading/trailing whitespace
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Step 2: ANNOTATOR  (gemini-3.5-flash)                     │
 │   system = tagset + rules ;  user = raw_text              │
 │   → tagged_text  (verbatim text + inline XML tags)        │
 └─────────────────────────────────────────────────────────┘
        │  raw_text ──────────────────────────┐ (forwarded ground-truth)
        ▼                                      ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Step 3: AUDITOR  (gpt-5.4-mini, cross-family)             │
 │   input = RAW_TEXT + ANNOTATED_TEXT                       │
 │   → JSON { status, error_type, auditor_reason }           │
 └─────────────────────────────────────────────────────────┘
        │
   ┌────┴───────────────────────────┐
   │ PASS                           │ FAIL
   ▼                                ▼
 Step 4A: deterministic regex     Step 4B: review queue
 parser (re.finditer)             (no automation)
   → clean_text + char spans         → row_id, raw, tagged,
   → verify clean==raw                  error_type, auditor_reason
   → gold_standard.csv +             → review_queue.csv
     gold_spans.csv                  (Step 5: human override CSV)
```

### Decision fork details

- **PASS → Step 4A (automated index generation).** The tagged text is swept left-to-right
  with compiled regex. On each open tag we record `label` + `start` (position in the
  progressively-built clean string); on the matching close tag we record `end` and slice the
  literal inner text. This counting is mathematical and sequential, so it cannot suffer the
  off-by-one / sub-token drift of generative coordinate emission. A defensive invariant check
  (`strip_tags(tagged) == raw_text`) re-routes any mismatch to the FAIL path even if the
  auditor passed it.
- **FAIL → Step 4B (HITL queue).** No automation. The row is written to `review_queue.csv`
  with the diagnostic hint so a human jumps straight to the disputed span.

---

## 4. Components & files

| File | Responsibility | Step |
|---|---|---|
| `pipeline/ingestion.py` | Stream CSV/JSONL rows; whitespace normalization | 1 |
| `pipeline/prompts.py` | Verbatim Stage-1 & Stage-2 system/user prompts | 2,3 |
| `pipeline/annotator.py` | Gemini call → `tagged_text` | 2 |
| `pipeline/auditor.py` | GPT call → `AuditResult` (structured) | 3 |
| `pipeline/schemas.py` | Pydantic `AuditResult`, `Span`, enums | 3,4 |
| `pipeline/parser.py` | Deterministic regex span/offset parser | 4A |
| `pipeline/writers.py` | CSV writers (gold, spans, review queue, run log) | 4 |
| `pipeline/orchestrator.py` | Per-row decision fork + batch driver | all |
| `run.py` | CLI entrypoint (`--input`, `--limit`, `--out-dir`, …) | — |
| `config.py` | Env-driven config (model names, keys, paths) | — |
| `tests/test_parser.py` | Parser correctness + invariant tests | 4A |

---

## 5. Data contracts

### Input (one record per row)
```jsonc
// reviews.jsonl
{ "id": "r0001", "text": "my 3yo loves this. perfect gift for a wife." }
```
CSV input: a `text` column (and optional `id` column; row index used if absent).

### Auditor structured output (`AuditResult`)
```jsonc
{
  "status": "PASS" | "FAIL",
  "error_type": "NONE" | "TEXT_MUTATION" | "INCORRECT_TAG" | "OVER_ANNOTATION"
              | "MISSING_TAG" | "WRONG_LABEL" | "WRONG_SPAN",
  "auditor_reason": "string (empty when PASS)"
}
```

### Outputs (all CSV, keyed by `row_id`)
- **`gold_standard.csv`** — `row_id, raw_text, tagged_text, num_entities, entities_json`
- **`gold_spans.csv`** (long/exploded) — `row_id, entity_index, label, text, start, end`
- **`review_queue.csv`** — `row_id, raw_text, tagged_text, error_type, auditor_reason`
- **`run_log.csv`** — `row_id, status, error_type, num_entities, note`

`start`/`end` are 0-based character offsets into `raw_text`, half-open
(`raw_text[start:end] == text`).

---

## 6. Prompts (source of truth)

Stage-1 and Stage-2 prompts are transcribed **verbatim from the PDF** into
`pipeline/prompts.py` (System–User decoupling: rules in the system message, the changing
review text isolated in the user message). Error-type taxonomy is fixed by the Stage-2 prompt:
`TEXT_MUTATION, INCORRECT_TAG, OVER_ANNOTATION, MISSING_TAG, WRONG_LABEL, WRONG_SPAN`.

---

## 7. Non-functional requirements

- **Streaming:** generator-based ingestion; never load the full corpus into memory.
- **Resilience:** retry with backoff on transient API errors; a row that errors hard is routed
  to the review queue with `error_type=PIPELINE_ERROR` rather than crashing the batch.
- **Resumability:** append-mode CSV writers + a processed-`row_id` set so a re-run skips
  already-committed rows.
- **Determinism:** annotator/auditor called at temperature 0 where supported.
- **Configurability:** model names, API keys, and paths via `.env` (doc defaults baked in).
- **Cost control:** each row = exactly 2 LLM calls (1 annotate + 1 audit); no retries on
  semantic disagreement (those go to humans).

---

## 8. Out of scope (v1)

- Postgres target DB (CSV chosen instead; schema is swap-compatible later).
- Rich web HITL UI (replaced by `review_queue.csv` + a human override CSV workflow).
- Active-learning / prompt auto-tuning loops.
- Multi-language support (English reviews assumed).

---

## 9. Open questions / assumptions

- Assumes input reviews are English and reasonably short (single-row reviews).
- `gemini-3.5-flash` / `gpt-5.4-mini` are treated as available model IDs; if the deployed
  IDs differ, override via `.env` (`GEMINI_MODEL`, `OPENAI_MODEL`).
- Human override step (Step 5) consumes `review_queue.csv` and produces a corrected
  `gold_spans.csv` append; the editing tool itself is a manual/CSV process in v1.
