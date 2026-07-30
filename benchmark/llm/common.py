"""
common.py

Backend-agnostic core of the GPT-NER-style benchmark, shared by both the Ollama
(ollama/annotate.py) and Hugging Face (hf/annotate.py) generation drivers.

Everything here is independent of HOW the tagged text is generated: the label
scheme, the annotation guideline default, prompt construction, kNN retrieval over
the datastore, output parsing/cleanup, and the resumable-CSV + comparison-table
machinery. Each backend's annotate.py adds only its own generation call and CLI.

LABEL SCHEME: this benchmark uses the collapsed 3-dimension scheme -- MINOR, GENDER,
KINSHIP -- from the naive guidelines (benchmark/guidelines/guideline_naive_3dim_*.txt),
not the pipeline's 5 fine labels. Model output is parsed for those 3 labels; retrieved
demonstrations (which carry the gold corpus's 5 fine tags) are remapped to the 3 coarse
tags before insertion; and predictions are scored against gold with its 5 labels folded
onto the same 3 dimensions (MINOR_AGE/MINOR_EDU->MINOR, GEN_NOUN/GEN_PHYS->GENDER,
FAM_KIN->KINSHIP).
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (COARSE_3DIM, FAILED_MARKER, THREE_DIM_LABELS,
                      load_gold, load_pred, print_report, score)

# pipeline/parser.py lives at the repo root (benchmark/llm/common.py -> repo root
# is three parents up); add it to the path so extract_entities can reuse the exact
# same deterministic tag parser the labeling pipeline itself uses.
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.parser import TagParseError, parse_tagged_text  # noqa: E402

# This benchmark targets the collapsed 3-dimension scheme (MINOR, GENDER, KINSHIP)
# used by benchmark/guidelines/guideline_naive_3dim_*.txt. Model output is parsed
# with these labels; the gold corpus's 5 fine labels are collapsed onto them (via
# COARSE_3DIM) both to remap retrieved demonstrations and to score against gold.
LABELS = THREE_DIM_LABELS

# Default guideline: the minimal 3-dimension prompt (overridable with --guideline-file).
DEFAULT_GUIDELINE_FILE = str(
    Path(__file__).resolve().parent.parent / "guidelines" / "guideline_naive_3dim_minimal.txt"
)

DEFAULT_TASK_INSTRUCTIONS = f"""You are an expert annotator. Tag spans in the input text that match one of
the following entity categories: {', '.join(LABELS)}.

Wrap each tagged span with an inline XML-style tag matching its label, e.g.:
"My <KINSHIP>great grandson</KINSHIP> loves this game."

Rules:
- Only tag spans that clearly match one of the categories above.
- Do not tag anything if no entities are present; return the text unchanged.
- Do not alter any text other than inserting the tags.
- Output ONLY the tagged text. No explanation, no preamble, no markdown fences.
"""


def remap_tagged_to_coarse(tagged_text: str) -> str:
    """Rewrite fine gold tags (<FAM_KIN>...) to their coarse 3-dim label (<KINSHIP>...).

    Retrieved demonstrations carry the datastore's 5 fine labels; remapping them keeps
    the few-shot examples consistent with the 3-dimension guideline the model follows.
    """
    def _repl(m: "re.Match") -> str:
        slash, label = m.group(1), m.group(2)
        return f"<{slash}{COARSE_3DIM.get(label, label)}>"
    return re.sub(r"<(/?)(" + "|".join(COARSE_3DIM) + r")>", _repl, tagged_text)


# ---------------------------------------------------------------------------
# Retrieval (sentence-level kNN, "Option 4")
# ---------------------------------------------------------------------------

class Datastore:
    def __init__(self, datastore_dir: str):
        ddir = Path(datastore_dir)
        self.embeddings = np.load(ddir / "datastore_embeddings.npy")  # (N, dim), already L2-normalized
        self.meta = pd.read_parquet(ddir / "datastore_meta.parquet")
        assert len(self.meta) == self.embeddings.shape[0], "Embeddings/metadata row count mismatch."

    def top_k(self, query_emb: np.ndarray, k: int) -> pd.DataFrame:
        """Cosine similarity via dot product (both sides pre-normalized)."""
        if k <= 0:
            return self.meta.iloc[0:0]
        sims = self.embeddings @ query_emb  # (N,)
        top_idx = np.argsort(-sims)[:k]
        result = self.meta.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(guideline_text: str, demos: pd.DataFrame, query_text: str) -> str:
    parts = [guideline_text.strip(), "", DEFAULT_TASK_INSTRUCTIONS.strip(), ""]

    if len(demos) > 0:
        parts.append("Examples:")
        for _, row in demos.iterrows():
            parts.append(f"Input: {row['raw_text']}")
            parts.append(f"Output: {remap_tagged_to_coarse(row['tagged_text'])}")
            parts.append("")

    parts.append("Now tag the following input.")
    parts.append(f"Input: {query_text}")
    parts.append("Output:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output parsing / validation
# ---------------------------------------------------------------------------

def extract_entities(tagged_text: str) -> list[dict]:
    """Recover an entities_json-style list of {label, text, start, end} from generated tagged text.

    Delegates to the pipeline's deterministic tag parser (pipeline/parser.py) so the
    benchmark uses the exact same offset-counting logic as the labeling pipeline itself.
    Unlike pipeline callers, benchmark generations come from an unaudited local model and
    aren't guaranteed well-formed, so malformed tag structure (unbalanced/mismatched) is
    swallowed into zero entities rather than raised.
    """
    try:
        _, spans = parse_tagged_text(tagged_text, tagset=LABELS)
    except TagParseError:
        return []
    return [{"label": s.label, "text": s.text, "start": s.start, "end": s.end} for s in spans]


def strip_malformed_tags(tagged_text: str) -> str:
    """Best-effort cleanup if the model emits a tag not in LABELS or an unclosed tag.
    Leaves well-formed tags alone; strips anything that doesn't match a known
    label by removing stray angle-bracket fragments.
    """
    known_open = {f"<{lab}>" for lab in LABELS} | {f"</{lab}>" for lab in LABELS}
    # remove any tag-like token not in the known set
    def _clean(match):
        tok = match.group(0)
        return tok if tok in known_open else ""
    return re.sub(r"</?[A-Z_]+>", _clean, tagged_text)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def condition_label(pred_csv_path: str, k_used: int | None = None) -> str:
    """'zero-shot (k=0)' / 'retriever (k=N)', reading k_used from the CSV if not given directly."""
    if k_used is None:
        with open(pred_csv_path, newline="", encoding="utf-8") as f:
            first = next(csv.DictReader(f), None)
        k_used = int(first["k_used"]) if first and "k_used" in first else None
    if k_used is None:
        return Path(pred_csv_path).stem
    return "zero-shot (k=0)" if k_used == 0 else f"retriever (k={k_used})"


def report_comparison(args: argparse.Namespace, df: pd.DataFrame, out_path: Path) -> None:
    """Print the with-retriever-vs-without-retriever comparison table (see annotate docstrings).

    No-ops (with a note) if --input-csv lacks gold entities_json, or --id-col isn't "row_id"
    (evaluate.py's scoring keys on that column name specifically).
    """
    if args.id_col != "row_id" or "entities_json" not in df.columns:
        if args.compare_with and "entities_json" not in df.columns:
            print("\nNote: --input-csv has no gold entities_json column; skipping --compare-with scoring.")
        return

    gold = load_gold(args.input_csv)
    print("\nScoring on 3 collapsed dimensions: " + ", ".join(LABELS)
          + " (gold's 5 fine labels folded via MINOR_AGE/EDU->MINOR, GEN_NOUN/PHYS->GENDER, FAM_KIN->KINSHIP).")
    own_name = condition_label(str(out_path), k_used=args.k)
    own_pred, own_failed = load_pred(str(out_path))
    own_result = score(gold, own_pred, own_failed, gold_label_map=COARSE_3DIM)
    print_report(own_name, own_result)

    if args.compare_with:
        other_name = condition_label(args.compare_with)
        other_pred, other_failed = load_pred(args.compare_with)
        other_result = score(gold, other_pred, other_failed, gold_label_map=COARSE_3DIM)
        print_report(other_name, other_result)

        print("\n=== WITH RETRIEVER vs WITHOUT RETRIEVER (micro-avg) ===")
        width = max(len(own_name), len(other_name))
        for name, result in [(own_name, own_result), (other_name, other_result)]:
            m = result["micro"]
            failed = result.get("failed_rows", 0)
            fail_note = f"  [{failed} failed row(s)]" if failed else ""
            print(f"  {name:<{width}}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}{fail_note}")


def load_progress(out_path: Path, id_col: str) -> tuple[set[str], set[str]]:
    """Row ids already written to a prior (possibly interrupted) --out-csv run.

    Returns (done_ids, failed_ids): done_ids is every id present in the file;
    failed_ids is the subset marked FAILED (all retries exhausted). Tolerates a
    missing, empty, or header-only file (all -> both empty) so a half-written file
    from a crash mid-flush doesn't break resume.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set(), set()
    done: set[str] = set()
    failed: set[str] = set()
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or id_col not in reader.fieldnames:
            return set(), set()
        has_pred = "predicted_entities_json" in reader.fieldnames
        for row in reader:
            rid = row[id_col]
            done.add(rid)
            if has_pred and row["predicted_entities_json"] == FAILED_MARKER:
                failed.add(rid)
    return done, failed


def drop_failed_rows(out_path: Path) -> int:
    """Rewrite --out-csv keeping only non-FAILED rows; return how many were dropped.

    Used when --retry-failed is set so previously-FAILED rows can be re-annotated
    and re-appended without leaving a duplicate id in the file.
    """
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    kept = [r for r in rows if r.get("predicted_entities_json") != FAILED_MARKER]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows) - len(kept)
