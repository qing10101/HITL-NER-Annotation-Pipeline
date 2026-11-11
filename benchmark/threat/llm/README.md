# LLM reconstruction attack (Scheme A, zero-shot Claude attacker)

Question: after IMPRINT's gold spans are removed from a review, how much of the removed
text can a frontier LLM guess back from the remaining context? This folder adds a
zero-shot Claude attacker to the Scheme A table next to the context-free prior and the
fine-tuned T5-base attackers in `benchmark/threat/`.

## Test set

`<repo>/test_balanced_400_human_validated - test_balanced_400.csv` (human-validated spans
in `tagged_text`). The 100 rows without a span are dropped: there is nothing to
reconstruct. Result: **300 reviews, 501 spans** (FAM_KIN 207, MINOR_AGE 165, GEN_NOUN 97,
GEN_PHYS 22, MINOR_EDU 10). The same 501 spans are written to
`benchmark/threat/data/test.csv` / `test_loo.csv`, so the prior and the T5 attackers are
re-scored on exactly the same spans (previous pipeline-labelled files kept as `*_pipeline.csv`;
4 rows differed).

## Conditions (identical span count, paired by span)

| condition | prompts | masks | what is hidden | question |
| --- | --- | --- | --- | --- |
| gold | 300 | 501 | every gold span | main result |
| loo | 320 (+181 re-used from gold) | 501 | one gold span at a time, the others visible | span-to-span leakage |
| rand (seeds 42, 43, 44) | 300 each | 501 each | same number of spans per row, same word length, content words only, never overlapping gold | are ordinary words just as guessable |

- The evaluation unit is the span, never the prompt. Each span is guessed exactly once per condition.
- LOO is expanded per span, so its count equals gold's by construction. In the 181
  single-span rows the LOO prompt is byte-identical to the gold prompt, so those
  predictions are copied from gold instead of being re-requested; the gold-vs-LOO
  contrast is reported on the 320 spans of the 119 multi-span rows (`paired.json`).
- Random masks use `common.random_spans(content_only=True)`: stopwords are excluded
  (otherwise `the`/`and` inflate the random condition), windows are length-matched to
  the gold span, and a shorter window is used only when a short review has no free
  window left. Seed 42 is the same set `recon_t5.py` scores. Three seeds give the
  sampling variance.
- Mask token `[MASK_k]` replaces the whole span: no length leak, no type leak, numbered
  left to right.

## Fairness rules

- **Policy-blind prompt** (`prompts.py`): the attacker is told only that some words were
  removed. This is the only prompt that is identical across gold, LOO and random masks
  (a policy-aware prompt would mislead the random condition). It makes the Claude row a
  lower bound on attacker strength; the policy-aware side is covered by the adaptive
  T5-base attacker, which was trained on gold-masked reviews.
- **Fresh context per review.** Every item is one API call that contains only the system
  prompt and that review. The attacker never sees gold labels, training data, other
  reviews, or this repository. The attack must not be run from an interactive Claude
  Code session that has read the test file.
- **Same scorer.** `common.normalize` and `data/lexicon.json` are the ones recon_t5.py uses:
  exact@1, exact@5 (five ranked guesses are requested), cat@1 (gold and LOO only; random
  spans have no type), anycat@1. Confidence is the model's self-reported 0-1 for its top
  guess, reported as `conf`; T5's teacher-forced gold probability is not observable
  through the API, so that column is left empty for the LLM.
- **Refusals and parse failures count as misses** (all five guesses empty) and are counted
  in `metrics.json`; no fallback to another model, which would mix attackers.
- **No sampling control.** Current models reject `temperature`; every request and response
  is cached under `raw/`, so the scored predictions are fixed and a rerun resumes
  without re-billing.
- **Uncertainty** is row-level bootstrap (spans in one review are correlated) plus an exact
  McNemar test for the paired contrasts.

## Memorisation check

Amazon Reviews 2023 is public. `memcheck.py` shows the model the gold-masked first half of
50 reviews and asks for the next sentence verbatim. Near-zero exact/near matches means the
attack numbers come from inference, not recall. Report the result in the paper either way.

## Run order

```powershell
# 0. once: API key
#    put ANTHROPIC_API_KEY=... in <repo>/.env (see .env.example) or set it in the environment

# 1. inputs (done; re-run if the validated CSV changes)
.venv-threat/Scripts/python benchmark/threat/llm/prepare.py

# 2. baselines on the same 501 spans (done once after prepare; ~10 min GPU)
.venv-threat/Scripts/python benchmark/threat/recon_t5.py --model t5-base --mode adaptive    --fp32 --tag fp32 --eval-only
.venv-threat/Scripts/python benchmark/threat/recon_t5.py --model t5-base --mode nonadaptive --fp32 --tag fp32 --eval-only
.venv-threat/Scripts/python benchmark/threat/prior_baseline.py

# 3. pilot: 20 reviews, prints per-label scores and token usage, writes preds_gold_pilot.csv
.venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond gold --limit 20

# 4. full grid (about 1,520 calls; gold must precede loo)
.venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond gold
.venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond loo
.venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond rand --seed 42
.venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond rand --seed 43
.venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond rand --seed 44

# 5. memorisation probe (50 calls)
.venv-threat/Scripts/python benchmark/threat/llm/memcheck.py

# 6. tables: metrics.json, paired.json, summary.md in the run directory
.venv-threat/Scripts/python benchmark/threat/llm/score.py
.venv-threat/Scripts/python benchmark/threat/summarize.py     # adds the row to results/summary.md
```

Flags: `--model` (default `claude-fable-5-1`; `claude-opus-5-5` is the cheaper alternative),
`--effort` (default `high`), `--workers` (default 4), `--dry-run` prints the first prompt.
Output directory: `benchmark/threat/results/recon_<model>_zeroshot_s0/`.

## Files

| file | role |
| --- | --- |
| `prepare.py` | validated CSV -> shared test split + `data/*.jsonl` + `manifest.json` |
| `prompts.py` | system/user prompt, JSON output schema, memorisation prompt |
| `run_attack.py` | API runner with per-item cache, LOO re-use, `preds_<cond>.csv` |
| `score.py` | `metrics.json`, `paired.json`, `summary.md` (prior + T5 + LLM in one table) |
| `memcheck.py` | continuation probe |
| `data/` | git-ignored inputs (review text) |

## Alternative without an API key: subagent dispatch (`dispatch.py`)

`dispatch.py make --cond <gold|loo|rand|memcheck> [--seed]` writes self-contained batch
prompts (default 15 reviews per batch) to `data/batches/<cond>/batch_NNN.txt`. Each batch is
handed verbatim to one fresh Claude Code subagent (not a fork, so it has none of the
orchestrating session's context) with the instruction to use no tools and reply with JSON
only. The reply is saved to `results/<run-dir>/raw_batches/<cond>/batch_NNN.json`, then
`dispatch.py collect --cond <cond>` scores it into `preds_<cond>.csv` and `score.py` works
as before.

Differences from the API path, to be stated in the paper:
- the model is whatever the session runs (Claude Fable 5.1 here), with harness defaults for
  effort and thinking; no temperature; reproducibility rests on the archived raw replies;
- 15 reviews share one context. No labels are ever in a batch, and the batch builder keeps
  two LOO items of the same review apart (they would reveal each other's span); gold, LOO
  and random prompts never mix, because the random prompt shows the gold spans in clear;
- the no-tools rule is enforced by instruction, not technically.

Batches: gold 20, LOO 24, random 20 per seed, memorisation 4 (108 dispatches).
