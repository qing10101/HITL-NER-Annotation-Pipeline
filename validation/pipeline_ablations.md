# Pipeline Ablations (C1–C4) on the Random-500 and Balanced-400 Sets, and Auditor vs Parser Rejections

## Evaluation sets

| Set | File | Rows | Spans | How it was sampled | Ground truth |
|---|---|---|---|---|---|
| Random 500 | `validation/general/sample_500_human.csv` | 500 | 156 | Uniform random sample of the 20k corpus | Independent human annotation |
| Balanced 400 | `validation/test_balanced_400_human_validated.csv` | 400 | 501 | 100 rows each with ≥1 minor entity, ≥1 gender entity, ≥1 kinship entity, and no entities | Human validation and correction of the pipeline's output |

The two sets share 10 rows.

- **Random 500** reflects the corpus as it is: 394 of its 500 rows (79%) have no entities. Because its annotators did not see
  the pipeline's output, its scores are an independent estimate.
- **Balanced 400** oversamples rows with entities, so it gives more spans per label. Its validators started
  from the pipeline's output, so its full-pipeline score is an upper bound.

**Predictions** for both sets come from the final corpus run in `output/`:
- Gemini's raw output ỹ from `annotator_cache.csv`;
- auditor (GPT-5.5) verdicts from `run_log.csv`;
- for escalated rows, the reviewer adjudication from `human_reviewed.csv`.

**Matching:** exact span on (label, start, end), micro-averaged over all rows of the set.

**Configurations:**

| Configuration | Ablation | Rows escalated to human review |
|---|---|---|
| Annotator only | C1 | none; offsets computed directly on ỹ, no `strip(ỹ) == x` check |
| Annotator + parser | C2 | rows where `parse_and_verify(ỹ, x)` fails |
| Annotator + auditor | C3 | rows the auditor FAILs; offsets as in C1 |
| Full pipeline | reference | auditor FAIL or parser failure |
| Same-family audit | C4 | rows a Gemini auditor FAILs (needs a new run, see section 3) |

In every configuration except annotator only, rows with no Gemini output (content-blocked) go to human
review, because the pipeline sends hard errors to review before any audit runs. Under annotator only, they
count as all misses.

## 1. Random 500

```
.venv/bin/python scripts/analysis/ablation_c1_c3.py --run-dir output \
    --human-csv validation/general/sample_500_human.csv
```

| Configuration | Rows escalated | Esc. rate | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| C1: annotator only | 0 | 0.0% | 148 | 6 | 8 | 0.9610 | 0.9487 | 0.9548 |
| C2: annotator + parser | 2 | 0.4% | 150 | 6 | 6 | 0.9615 | 0.9615 | 0.9615 |
| C3: annotator + auditor | 6 | 1.2% | 153 | 6 | 3 | 0.9623 | 0.9808 | 0.9714 |
| Full pipeline | 6 | 1.2% | 153 | 6 | 3 | 0.9623 | 0.9808 | 0.9714 |
| C4: same-family audit | — | — | — | — | — | — | — | — (not run yet) |

### Per-label F1

| Label | C1 | C2 | C3 | Full pipeline |
|---|---|---|---|---|
| MINOR_AGE | 0.9259 | 0.9358 | 0.9550 | 0.9550 |
| MINOR_EDU | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| GEN_NOUN | 0.9412 | 0.9412 | 0.9412 | 0.9412 |
| GEN_PHYS | 0.9333 | 0.9333 | 1.0000 | 1.0000 |
| FAM_KIN | 0.9759 | 0.9820 | 0.9820 | 0.9820 |

### Findings

- **Escalations:** the auditor failed 6 rows (1.2%). 4 were OMITTED_VALID_TAG. The other two are the
  2 rows the parser rejects:
  - `B08TGMTXG4_69328`: Gemini returned no output (PIPELINE_ERROR). Human review recovers its 2
    entities, which is all of C2's gain over C1.
  - `B00AUZS5LG_87439`: Gemini changed `&#34;` to `"`. The auditor labelled this RAW_TEXT_MUTATION. The
    row has no entities, so the change caused no offset errors.
- **C3 equals the full pipeline**, as on the balanced set. Both parser-rejected rows were already escalated.
- **Most remaining errors come from rows the auditor passed** (6 FP, 3 FN in 7 rows). In these rows the
  pipeline and the independent annotators disagree on borderline guideline cases:
  - `baby` tagged MINOR_AGE in 3 rows where the human annotators left it untagged;
  - `2` tagged MINOR_AGE (`B00U770P24_133077`);
  - `little girl` tagged MINOR_AGE where the human annotators tagged `girl` as FAM_KIN
    (`1582381291_93797`);
  - `grandmas` labelled FAM_KIN where the human annotators labelled it GEN_NOUN (`B093CZ17JS_97886`);
  - `children` as FAM_KIN, missed (`0399158561_71394`).
- **Low entity density:** with 156 spans across 500 rows, each span moves recall by about 0.6 points.

### Comparison with the earlier validation run

`validation/ablation_c1_c3.md` scores the same 500 rows against an earlier pipeline run
(`--run-dir validation/general`), made before the corpus run:

| Run | C1 F1 | C2 F1 | C3 / Full F1 | Escalated |
|---|---|---|---|---|
| Earlier run (`validation/general`) | 0.9389 | 0.9457 | 0.9747 | 13 (2.6%) |
| Corpus run (`output/`) | 0.9548 | 0.9615 | 0.9714 | 6 (1.2%) |

Gemini's output differs between the two runs in 9 of the 500 rows. In 6 of the 9, the corpus run is
correct and the earlier run is wrong. The corpus run's auditor escalated fewer rows: 6 instead of 13. Its
annotator was more accurate, so the auditor had less to catch, and the final F1 is about the same (0.971
vs 0.975).

## 2. Balanced 400

```
.venv/bin/python scripts/analysis/ablation_c1_c3.py --run-dir output \
    --human-csv validation/test_balanced_400_human_validated.csv --raw-col raw_text --gold-col tagged_text
```

| Configuration | Rows escalated | Esc. rate | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| C1: annotator only | 0 | 0.0% | 466 | 14 | 35 | 0.9708 | 0.9301 | 0.9501 |
| C2: annotator + parser | 10 | 2.5% | 484 | 10 | 17 | 0.9798 | 0.9661 | 0.9729 |
| C3: annotator + auditor | 30 | 7.5% | 498 | 3 | 3 | 0.9940 | 0.9940 | 0.9940 |
| Full pipeline | 30 | 7.5% | 498 | 3 | 3 | 0.9940 | 0.9940 | 0.9940 |
| C4: same-family audit | — | — | — | — | — | — | — | — (not run yet) |

### Per-label F1

| Label | C1 | C2 | C3 | Full pipeline |
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
  - 6 TP and the 4 FP come from the 6 rows where Gemini changed the text or broke a tag (see section 4).
- **Annotator + auditor equals the full pipeline.** The auditor had already failed every row the parser
  rejected, so adding the parser changes nothing here.
- **The full pipeline's 3 FP and 3 FN are exactly the validators' 4 edits:**
  - added `kids` as FAM_KIN (`B087CBJ465_40549`);
  - widened `Daughter` to `Daughter s` (`B073WJ44RY_151526`);
  - relabelled `grandmother` from FAM_KIN to GEN_NOUN (`B07MPRX14B_07630`);
  - removed the `kiddos` MINOR_AGE tag (`B07RDL3PMJ_31281`).

  The validators started from the pipeline's output, so 0.994 is an upper bound, not an independent
  estimate. The random-500 score (0.971) is the independent estimate.

### Assumptions

- **Content-blocked rows:** `B01BRVY6TG_46552`, `B004NCEL4M_40529`, `B09VTZ1V5C_179053` and
  `B0BTFY3V69_89442` have no Gemini output. Under annotator only they count as all misses (12 FN). In the
  other three configurations they go to human review. C3's 30 escalations are therefore 26 auditor FAILs
  plus these 4 rows.

## 3. C4: same-family audit (Gemini auditing Gemini)

C4 is the only ablation that needs new API calls, because the corpus run's auditor was GPT-5.5.
`scripts/analysis/ablation_c4.py` re-audits the cached Gemini output ỹ (unchanged from C1–C3) with a Gemini
auditor, using the pipeline's own Auditor and prompt. It then scores three configurations: cross-family
auditor only (equal to C3), same-family auditor only, and same-family auditor plus parser. Results: pending.

## 4. Auditor vs parser rejections

### Balanced 400

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

### Random 500

The parser rejects 2 rows; the auditor failed both. `B00AUZS5LG_87439` (`&#34;` → `"`) was correctly
labelled RAW_TEXT_MUTATION. `B08TGMTXG4_69328` had no Gemini output.

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
