# Pipeline Ablations C1–C3 (validation set, 500 reviews)

Scored against the independent human annotation (`validation/general/sample_500_human.csv`) using the
artifacts of the existing run in `validation/general/` (no new API calls). Reproduce with:

```
.venv/bin/python scripts/analysis/ablation_c1_c3.py
```

**Matching:** exact span on (label, start, end), micro-averaged over all 500 rows (156 ground-truth spans).
**Escalated rows** take the real reviewer adjudication from `human_reviewed.csv`. Every row escalated under
C2 or C3 was also in the full pipeline's review queue, so no oracle substitution was needed.

## Results

| Condition | Escalated | Esc. rate | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Full pipeline (reference) | 13 | 2.6% | 154 | 6 | 2 | 0.9625 | 0.9872 | 0.9747 |
| C1: raw Gemini, no parser check | 0 | 0.0% | 146 | 9 | 10 | 0.9419 | 0.9359 | 0.9389 |
| C2: parser-only escalation | 2 | 0.4% | 148 | 9 | 8 | 0.9427 | 0.9487 | 0.9457 |
| C3: auditor-only escalation | 13 | 2.6% | 154 | 6 | 2 | 0.9625 | 0.9872 | 0.9747 |
| C4: same-family audit | — | — | — | — | — | — | — | — (not run yet; see `scripts/analysis/ablation_c4.py`) |

The full-pipeline row matches `validation/diff-general.txt` (154 exact matches, 6 gold-only, 2 human-only).

### Per-label F1

| Label | Full | C1 | C2 | C3 |
|---|---|---|---|---|
| MINOR_AGE | 0.9369 | 0.9091 | 0.9189 | 0.9369 |
| MINOR_EDU | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| GEN_NOUN | 1.0000 | 0.9474 | 0.9474 | 1.0000 |
| GEN_PHYS | 1.0000 | 0.8000 | 0.8000 | 1.0000 |
| FAM_KIN | 0.9940 | 0.9693 | 0.9756 | 0.9940 |

## Findings

- **No offset drift on this set.** Raw Gemini output changed the text in only one row (`B00AUZS5LG_87439`,
  `&#34;` → `"`), and that row has no entities, so it caused no offset errors. No rows had broken or
  mismatched tags.
- **C2's gain over C1 comes from one row.** Gemini returned no output for `B08TGMTXG4_69328`. The parser
  check escalates that row, and human review recovers its 2 entities.
- **C3 equals the full pipeline.** Both parser-rejected rows were already escalated: one failed the audit,
  and the other errored before the audit ran. All accuracy gains over C1 come from the auditor's semantic
  catches (omitted tags, out-of-scope tags, unanchored tags, span boundaries).
- On this set the parser check acts as a safety net with no measurable accuracy contribution.

## Assumptions

- **C1 / C3 parsing:** offsets are computed directly on ỹ with `parse_tagged_text`, with no
  `strip(ỹ) == x` check.
- **Empty annotator output** (`B08TGMTXG4_69328`, PIPELINE_ERROR): counts as all misses in C1, fails the
  parser check in C2, and is escalated in C3 because the pipeline sends hard errors to review before any
  audit. C3's 13 escalations are therefore 12 auditor FAILs plus this 1 error row. If the row is excluded
  from C3's escalation, C3 has 12 escalations (2.4%), recall 0.9744 and F1 0.9684.
- **Ground-truth text:** one ground-truth row, `B09CDDRHHJ_02441`, doesn't exactly reproduce the raw text.
  The human annotation drops its foreign-language (Spanish) sentence, which is out of scope for
  annotation. The row has no entities, so it doesn't affect the scores. (A whitespace mismatch in
  `B087F3H31V_18033` has been fixed in the ground truth and didn't change any numbers.)
