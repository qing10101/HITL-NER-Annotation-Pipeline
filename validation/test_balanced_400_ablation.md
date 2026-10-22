# Ablations on the Balanced Test Set (400 rows), and Auditor vs Parser Rejections

## 1. Span-level P/R/F1 by pipeline configuration

**Data**
- **Ground truth:** `validation/test_balanced_400_human_validated.csv` (400 rows, 501 spans), the
  human-validated version of `output/test_balanced_400.csv`.
- **Predictions:** taken from the final corpus run in `output/`:
  - Gemini's raw output from `annotator_cache.csv`;
  - auditor verdicts from `run_log.csv`;
  - for escalated rows, the reviewer adjudication from `human_reviewed.csv`.

**Matching:** exact span on (label, start, end), micro-averaged.

```
.venv/bin/python scripts/analysis/ablation_c1_c3.py --run-dir output \
    --human-csv validation/test_balanced_400_human_validated.csv --raw-col raw_text --gold-col tagged_text
```

| Configuration | Rows escalated | Esc. rate | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Annotator only | 0 | 0.0% | 466 | 14 | 35 | 0.9708 | 0.9301 | 0.9501 |
| Annotator + parser | 10 | 2.5% | 484 | 10 | 17 | 0.9798 | 0.9661 | 0.9729 |
| Annotator + auditor | 30 | 7.5% | 498 | 3 | 3 | 0.9940 | 0.9940 | 0.9940 |
| Full pipeline | 30 | 7.5% | 498 | 3 | 3 | 0.9940 | 0.9940 | 0.9940 |

### Per-label F1

| Label | Annotator only | + parser | + auditor | Full pipeline |
|---|---|---|---|---|
| MINOR_AGE | 0.9467 | 0.9755 | 0.9970 | 0.9970 |
| MINOR_EDU | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| GEN_NOUN | 0.9789 | 0.9843 | 0.9948 | 0.9948 |
| GEN_PHYS | 0.8444 | 0.9091 | 1.0000 | 1.0000 |
| FAM_KIN | 0.9484 | 0.9710 | 0.9903 | 0.9903 |

Entity counts: FAM_KIN 207, MINOR_AGE 166, GEN_NOUN 96, GEN_PHYS 22, MINOR_EDU 10. With only 22 and 10
entities, the per-label F1 for GEN_PHYS and MINOR_EDU moves several points per span.

### Findings

- **The parser's gain over annotator only is 18 TP and 4 fewer FP:**
  - 12 TP come from the 4 content-blocked rows. Gemini returned no output for them, so the parser check
    sends them to review.
  - 6 TP and the 4 FP come from the 6 rows where Gemini changed the text or broke a tag (see section 2).
- **Annotator + auditor equals the full pipeline.** The auditor had already failed every row the parser
  rejected, so adding the parser changes nothing here.
- **The full pipeline's 3 FP and 3 FN are exactly the validators' 4 edits:**
  - added `kids` as FAM_KIN (`B087CBJ465_40549`);
  - widened `Daughter` to `Daughter s` (`B073WJ44RY_151526`);
  - relabelled `grandmother` from FAM_KIN to GEN_NOUN (`B07MPRX14B_07630`);
  - removed the `kiddos` MINOR_AGE tag (`B07RDL3PMJ_31281`).

  The validators started from the pipeline's output, so 0.994 is an upper bound, not an independent
  estimate.

### Assumptions

- **Content-blocked rows:** `B01BRVY6TG_46552`, `B004NCEL4M_40529`, `B09VTZ1V5C_179053` and
  `B0BTFY3V69_89442` have no Gemini output. They count as all misses (12 FN) under annotator only. In the
  other three configurations they go to human review, because the pipeline sends hard errors to review
  before any audit runs.
- **Tag typo:** `B07MPRX14B_07630` in the validated file has the typo `<GEM_NOUN>grandmother</GEN_NOUN>`.
  It was scored as GEN_NOUN using a corrected copy. The command above, run on the file as it is, will
  treat that row as having no entities until the typo is fixed.

## 2. Auditor vs parser rejections

### Test set (400 rows)

| | Rows | % of 400 |
|---|---|---|
| In review queue | 30 | 7.5% |
| ...auditor FAIL | 26 | 6.5% |
| ...parser FAIL after an auditor PASS | 0 | 0.0% |
| ...content-blocked (no Gemini output) | 4 | 1.0% |
| Parser rejects raw Gemini output (all 396 annotated rows) | 6 | 1.5% |

The auditor had already failed all 6 parser-rejected rows, so 6 of the 26 auditor FAILs (23%) would also
fail the parser. **None of the 6 was labelled RAW_TEXT_MUTATION:**

| Row | What the parser found | Auditor's error type |
|---|---|---|
| `B0096DCG6I_36692` | Malformed tag: `son</_FAM_KIN>` | INVALID_SPAN_BOUNDARY |
| `B08J3QP3JX_113841` | 3 inserted spaces | INVALID_SPAN_BOUNDARY |
| `B006U32EUK_33241` | 7 inserted spaces | OUT_OF_SCOPE_TAG |
| `B0827F5VLB_17527` | Repaired garbled characters (`â€™` → `’`) plus extra spaces, 11 changes in all | UNANCHORED_TAGGING |
| `B00FZMDAO6_27853` | 2 repaired garbled characters (`â€™` → `’`) | OMITTED_VALID_TAG |
| `B00XFCJYP2_38672` | 7 inserted spaces | OMITTED_VALID_TAG |

- **The malformed tag (1 row):** the auditor saw this one. Its reason quotes `</_FAM_KIN>`, but it filed
  the row as a span-boundary error.
- **The changed text (5 rows):** the auditor didn't notice. Its prompt ranks RAW_TEXT_MUTATION above every
  other error type ("DOMINANT ERROR SELECTION" in `pipeline/prompts.py`), so it would have reported the
  change if it had seen it. Instead, each reason describes an unrelated tagging problem.

The auditor covering every parser rejection on this set is therefore a coincidence. Those rows also had a
separate tagging error, and that is what the auditor caught.

### Full corpus (20,000 rows)

Gemini changed the text or broke a tag in 69 rows. The auditor handled them as follows:

| What the auditor did | Rows | Share |
|---|---|---|
| Flagged RAW_TEXT_MUTATION | 18 | 26% |
| Failed the row for a different reason (the change went unnoticed) | 47 | 68% |
| Passed the row (caught only by the parser) | 4 | 6% |

**The 47 rows where the change went unnoticed:**
- extra spaces, mostly a second space after a sentence: 28 rows;
- repaired garbled characters (`â€™` → `’`, `â€œ` → `“`): 16 rows;
- malformed closing tags (`</_FAM_KIN>`, `</_KIN>`, `</_num>`): 3 rows.

**The 4 rows the auditor passed:**
- `B07CCG2VSY_163243`: the word "invalid" was deleted from "my invalid mother". This is a change in
  meaning, and the deleted word arguably describes the mother's health.
- `B09P2LTV45_106691`: a `<br />` line break was replaced with a space.
- `B0080X23D6_108621` and `B08SQP62ZW_30882`: a trailing `"` was dropped.

**Takeaway:** the auditor is unreliable at noticing changed text, especially whitespace and encoding
fixes. It recognised only 18 of the 69 changed-text rows. Every changed-text row still went to review,
but only because the auditor failed it for an unrelated reason or the parser caught it. The parser check
is the dependable safeguard for preserving the original text. Its contribution to F1 is near zero on these
sets only because almost every changed-text row also had a tagging error that the auditor caught.
