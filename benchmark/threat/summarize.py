"""Aggregate benchmark/threat/results/*/metrics.json into results/summary.md (+ summary.json).

    python benchmark/threat/summarize.py

Scheme B rows:  presence_<model>_<train-on>_s<seed>   (conditions raw / gold_del / rand_del)
Scheme A rows:  recon_<model>_<mode>_s<seed>          (conditions gold / loo / rand)
Several seeds of the same run are averaged (mean, with std in parentheses when n>1).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
B_LABELS = ["minor", "gender", "kinship"]
B_CONDS = ["raw", "gold_del", "rand_del"]
A_CONDS = ["gold", "loo", "rand"]
A_LABELS = ["all", "FAM_KIN", "MINOR_AGE", "MINOR_EDU", "GEN_NOUN", "GEN_PHYS", "RANDOM"]


def fmt(vals: list[float | None]) -> str:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{np.mean(v):.3f} ({np.std(v):.3f})"


def collect(prefix: str) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for d in sorted(RESULTS.glob(f"{prefix}_*_s*")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        key = re.sub(r"_s\d+$", "", d.name)
        runs[key].append(json.loads(f.read_text(encoding="utf-8")))
    return runs


def scheme_b(runs) -> list[str]:
    out = ["## Scheme B: dimension-presence attacker", "",
           "AUROC per label, macro AUROC, mean probability on positives (p+), attack success "
           "(p>=0.8 and positive). Train-on `raw` = non-adaptive, `gold_del` = adaptive.", "",
           "| attacker | test cond | minor AUROC | gender AUROC | kinship AUROC | macro AUROC | macro F1 | p+ minor | p+ gender | p+ kinship | succ minor | succ gender | succ kinship |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, ms in runs.items():
        name = key.replace("presence_", "")
        for cond in B_CONDS:
            if not all(cond in m for m in ms):
                continue
            cells = [name + f" (n={len(ms)})" if len(ms) > 1 else name, cond]
            cells += [fmt([m[cond][l]["auroc"] for m in ms]) for l in B_LABELS]
            cells += [fmt([m[cond]["macro"]["auroc"] for m in ms]), fmt([m[cond]["macro"]["f1"] for m in ms])]
            cells += [fmt([m[cond][l]["mean_p_pos"] for m in ms]) for l in B_LABELS]
            cells += [fmt([m[cond][l]["attack_success_pos"] for m in ms]) for l in B_LABELS]
            out.append("| " + " | ".join(cells) + " |")
    # residual cues from the tf-idf runs
    for d in sorted(RESULTS.glob("presence_tfidf_*_s*")):
        c = d / "cues.json"
        if c.exists():
            cues = json.loads(c.read_text(encoding="utf-8"))
            out += ["", f"**Top positive word features, {d.name}** (logistic-regression coefficients)", ""]
            for l in B_LABELS:
                out.append(f"- {l}: " + ", ".join(f"{w} ({v})" for w, v in cues[l][:20]))
    return out


def scheme_a(runs) -> list[str]:
    out = ["", "## Scheme A: span reconstruction attacker", "",
           "exact@1/@5 = normalised prediction equals the masked span; cat@1 = top prediction falls in the "
           "same entity type's lexicon; anycat@1 = in any privacy lexicon; gold p = teacher-forced probability "
           "of the gold span. `nonadaptive` never saw a privacy span (trained on random spans of negative rows).", "",
           "Caveat: the `t5gemma2_*_fp32` attackers learned little beyond the training-span prior (context swap changes "
           "gold-span probability by a log-ratio of 0.2 vs 3.3 for t5-base; see docs/threat_experiment_design.md, section 8); "
           "`t5-base_*_fp32` is the usable reconstruction attacker. Earlier bf16 runs are invalid and kept in results/invalid_bf16/.", "",
           "| attacker | cond | label | n | exact@1 | exact@5 | cat@1 | anycat@1 | gold p |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, ms in runs.items():
        name = key.replace("recon_", "")
        for cond in A_CONDS:
            if not all(cond in m for m in ms):
                continue
            for lab in A_LABELS:
                if not all(lab in m[cond] for m in ms):
                    continue
                g = [m[cond][lab] for m in ms]
                out.append("| " + " | ".join([
                    name + (f" (n={len(ms)})" if len(ms) > 1 else ""), cond, lab, str(g[0]["n"]),
                    fmt([x["exact1"] for x in g]), fmt([x["exact5"] for x in g]),
                    fmt([x["cat1"] for x in g]), fmt([x["anycat1"] for x in g]),
                    fmt([x["gold_prob"] for x in g])]) + " |")
    pb = RESULTS / "prior_baseline.json"
    if pb.exists():  # context-free reference rows (benchmark/threat/prior_baseline.py)
        P = json.loads(pb.read_text(encoding="utf-8"))["test"]
        for lab, v in P.items():
            out.append(f"| context-free prior (always top training spans) | any | {lab} | {v['n']} | "
                       f"{v['prior_exact1']:.3f} | {v['prior_exact5']:.3f} | {v['prior_cat1']:.3f} | – | – |")
        for lab, v in P.items():
            out.append(f"| type-oracle prior (top spans of the true type) | any | {lab} | {v['n']} | "
                       f"{v['oracle_exact1']:.3f} | {v['oracle_exact5']:.3f} | 1.000 | – | – |")
    return out


def main() -> None:
    b, a = collect("presence"), collect("recon")
    lines = ["# Threat experiment results", "",
             f"Runs found: {len(b)} presence, {len(a)} recon (seeds averaged where repeated).", ""]
    lines += scheme_b(b) + scheme_a(a)
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps({"presence": b, "recon": a}, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
