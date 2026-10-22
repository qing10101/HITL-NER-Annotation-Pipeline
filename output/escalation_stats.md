# Final Corpus: Auditor and Parser Escalation

Computed from `output/run_log.csv`, `output/review_queue.csv` and `output/annotator_cache.csv`
(20,000 rows, all row_ids unique).

## Routing outcome (as run)

Each row has exactly one outcome.

| Outcome | Rows | % of 20,000 |
|---|---|---|
| Committed (auditor PASS and parser PASS) | 19,574 | 97.87% |
| **Auditor FAIL** | **382** | **1.91%** |
| **Parser FAIL** (auditor PASS, rejected by the parser backstop) | **4** | **0.02%** |
| Annotator error, no output (39 content-blocked, 1 timeout) | 40 | 0.20% |
| **Total escalated to human review** | **426** | **2.13%** |

All 4 parser rejections failed the character-preservation check (`strip(ỹ) != x`).

### Auditor FAIL by error type

| Error type | Rows | % of 20,000 |
|---|---|---|
| OMITTED_VALID_TAG | 191 | 0.96% |
| OUT_OF_SCOPE_TAG | 80 | 0.40% |
| UNANCHORED_TAGGING | 49 | 0.25% |
| INVALID_SPAN_BOUNDARY | 36 | 0.18% |
| RAW_TEXT_MUTATION | 18 | 0.09% |
| MISALLOCATED_LABEL | 7 | 0.04% |
| NON_HUMAN_TAGGING | 1 | 0.01% |
| **Total** | **382** | **1.91%** |

## Parser check across all annotator outputs

The table above only counts the parser where it overrode an auditor PASS. Running the parser check on all
19,960 rows with annotator output shows how many rows it would reject on its own:

| | Rows | % of 20,000 | % of 19,960 annotated |
|---|---|---|---|
| Malformed tags (unbalanced / mismatched / unclosed) | 12 | 0.06% | 0.06% |
| Character-preservation violation | 57 | 0.29% | 0.29% |
| **Total parser rejections** | **69** | **0.35%** | **0.35%** |
| ...also failed by the auditor | 65 | 0.33% | 0.33% |
| ...caught only by the parser | 4 | 0.02% | 0.02% |

The auditor independently caught 65 of the 69 parser rejections (94%). Of the auditor's 382 FAILs, 317
(1.59% of the corpus) were structurally valid and could only be caught by the auditor's semantic checks.
