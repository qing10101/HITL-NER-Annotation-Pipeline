"""Scoring for the LLM reconstruction attacker.

    .venv-threat/Scripts/python benchmark/threat/llm/score.py [--run recon_claude-fable-5-1_zeroshot_s0]

Writes into the run directory:
    metrics.json   per condition x label, same shape recon_t5.py writes (summarize.py reads it)
    paired.json    paired contrasts on identical span sets:
                     gold vs loo   on the 320 spans of multi-span rows (single-span rows are identical prompts)
                     gold vs rand  per random seed, paired by (row_id, k-th mask of the row)
                   with row-level bootstrap 95% CIs (spans in one review are not independent)
                   and an exact McNemar test on exact@1
    summary.md     one table: context-free prior, type-oracle prior, T5-base attackers, this run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THREAT = HERE.parent
sys.path.insert(0, str(THREAT))
from common import LABELS  # noqa: E402

RESULTS = THREAT / "results"
A_LABELS = ["all"] + LABELS + ["RANDOM"]


def summarize(pred: pd.DataFrame) -> dict:
    out = {}
    for lab in A_LABELS:
        g = pred if lab == "all" else pred[pred.label == lab]
        if g.empty:
            continue
        out[lab] = {"n": int(len(g)),
                    "exact1": float(g.exact1.mean()), "exact5": float(g.exact5.mean()),
                    "cat1": (float(g.cat1.mean()) if g.cat1.notna().any() else None),
                    "anycat1": float(g.anycat1.mean()),
                    "gold_prob": None,
                    "conf1": float(g.conf1.mean()),
                    "conf1_when_right": (float(g[g.exact1 == 1].conf1.mean()) if (g.exact1 == 1).any() else None),
                    "conf1_when_wrong": (float(g[g.exact1 == 0].conf1.mean()) if (g.exact1 == 0).any() else None),
                    "refusals": int((g.stop_reason == "refusal").sum()) if "stop_reason" in g else 0}
    return out


def row_bootstrap(df: pd.DataFrame, col: str, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in df.groupby("row_id")]
    sizes = np.array([len(g) for g in groups])
    sums = np.array([g.sum() for g in groups])
    idx = rng.integers(0, len(groups), size=(reps, len(groups)))
    est = sums[idx].sum(1) / sizes[idx].sum(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], name_a: str, name_b: str) -> dict:
    m = a.merge(b, on=keys, suffixes=(f"_{name_a}", f"_{name_b}"))
    res = {"n": int(len(m))}
    for metric in ("exact1", "exact5", "conf1"):
        ca, cb = f"{metric}_{name_a}", f"{metric}_{name_b}"
        m["_d"] = m[ca] - m[cb]
        lo, hi = row_bootstrap(m, "_d")
        res[metric] = {name_a: float(m[ca].mean()), name_b: float(m[cb].mean()),
                       "diff": float(m["_d"].mean()), "ci95": [lo, hi]}
        if metric != "conf1":
            bb = int(((m[ca] == 1) & (m[cb] == 0)).sum())
            cc = int(((m[ca] == 0) & (m[cb] == 1)).sum())
            res[metric]["mcnemar_b_c"] = [bb, cc]
            res[metric]["mcnemar_p"] = mcnemar_exact(bb, cc)
    return res


def write_metrics(run_dir: Path) -> dict:
    metrics = {}
    for f in sorted(run_dir.glob("preds_*.csv")):
        if f.stem.endswith("_pilot"):
            continue
        cond = f.stem[len("preds_"):]
        pred = pd.read_csv(f, encoding="utf-8")
        metrics[cond] = summarize(pred)
        metrics[cond]["all"]["exact1_ci95"] = list(row_bootstrap(pred, "exact1"))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def write_paired(run_dir: Path) -> dict:
    preds = {f.stem[len("preds_"):]: pd.read_csv(f, encoding="utf-8")
             for f in run_dir.glob("preds_*.csv") if not f.stem.endswith("_pilot")}
    out = {}
    if "gold" in preds and "loo" in preds:
        loo_only = preds["loo"][preds["loo"].item_id != "reuse:gold"]
        out["gold_vs_loo_multispan"] = paired(preds["gold"], loo_only, ["row_id", "span_idx"], "gold", "loo")
        out["gold_vs_loo_all"] = paired(preds["gold"], preds["loo"], ["row_id", "span_idx"], "gold", "loo")
    for cond in sorted(k for k in preds if k.startswith("rand")):
        if "gold" in preds:
            g = preds["gold"].copy()
            r = preds[cond].copy()
            g["k"] = g.groupby("row_id").cumcount()
            r["k"] = r.groupby("row_id").cumcount()
            out[f"gold_vs_{cond}"] = paired(g, r, ["row_id", "k"], "gold", "rand")
    (run_dir / "paired.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def fmt(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def write_summary(run_dir: Path, metrics: dict, pairs: dict) -> str:
    lines = [f"# LLM reconstruction attack: {run_dir.name}", "",
             "Unit = masked span (501 in every condition). exact@1/@5 = normalised top-1 / any-of-5 guess equals "
             "the removed text; cat@1 = top guess in the span type's lexicon; anycat@1 = in any privacy lexicon; "
             "conf = self-reported confidence in the top guess (not comparable to T5's teacher-forced gold p).", "",
             "| attacker | cond | label | n | exact@1 | exact@5 | cat@1 | anycat@1 | conf / gold p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    pb = RESULTS / "prior_baseline.json"
    if pb.exists():
        P = json.loads(pb.read_text(encoding="utf-8"))["test"]
        for lab, v in P.items():
            lines.append(f"| context-free prior | any | {lab} | {v['n']} | {v['prior_exact1']:.3f} | {v['prior_exact5']:.3f} | {v['prior_cat1']:.3f} | – | – |")
        for lab, v in P.items():
            lines.append(f"| type-oracle prior | any | {lab} | {v['n']} | {v['oracle_exact1']:.3f} | {v['oracle_exact5']:.3f} | 1.000 | – | – |")
    for d in sorted(RESULTS.glob("recon_t5-base_*_fp32_s42")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        M = json.loads(f.read_text(encoding="utf-8"))
        name = d.name.replace("recon_", "").replace("_fp32_s42", "")
        for cond in ("gold", "loo", "rand"):
            for lab in A_LABELS:
                v = M.get(cond, {}).get(lab)
                if v:
                    lines.append(f"| {name} | {cond} | {lab} | {v['n']} | {fmt(v['exact1'])} | {fmt(v['exact5'])} | {fmt(v['cat1'])} | {fmt(v['anycat1'])} | {fmt(v['gold_prob'])} |")
    name = run_dir.name.replace("recon_", "").replace("_s0", "")
    for cond, labs in metrics.items():
        for lab in A_LABELS:
            v = labs.get(lab)
            if v:
                lines.append(f"| {name} | {cond} | {lab} | {v['n']} | {fmt(v['exact1'])} | {fmt(v['exact5'])} | {fmt(v['cat1'])} | {fmt(v['anycat1'])} | {fmt(v['conf1'])} |")
    if pairs:
        lines += ["", "## Paired contrasts (row-level bootstrap 95% CI, exact McNemar on exact@1)", "",
                  "| contrast | n | metric | A | B | A−B | CI95 | McNemar b/c | p |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, v in pairs.items():
            for metric in ("exact1", "exact5", "conf1"):
                r = v[metric]
                a, b = [k for k in r if k not in ("diff", "ci95", "mcnemar_b_c", "mcnemar_p")]
                bc = r.get("mcnemar_b_c")
                lines.append(f"| {key} | {v['n']} | {metric} | {r[a]:.3f} | {r[b]:.3f} | {r['diff']:+.3f} | "
                             f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {bc[0]}/{bc[1] if bc else '–'} | {fmt(r.get('mcnemar_p'))} |"
                             if bc else
                             f"| {key} | {v['n']} | {metric} | {r[a]:.3f} | {r[b]:.3f} | {r['diff']:+.3f} | "
                             f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | – | – |")
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run directory name under results/ (default: newest recon_claude*)")
    args = ap.parse_args()
    if args.run:
        run_dir = RESULTS / args.run
    else:
        cands = sorted(RESULTS.glob("recon_claude*_s0"), key=lambda p: p.stat().st_mtime)
        if not cands:
            sys.exit("no recon_claude*_s0 run found under results/")
        run_dir = cands[-1]
    metrics = write_metrics(run_dir)
    pairs = write_paired(run_dir)
    text = write_summary(run_dir, metrics, pairs)
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
