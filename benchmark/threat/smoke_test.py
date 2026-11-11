"""Step 1 smoke test: can the chosen 2025 models run on this machine, and does
T5Gemma emit sentinel-format span fills zero-shot?

Run inside .venv-threat:
    python benchmark/threat/smoke_test.py [--n 20] [--skip-neobert] [--skip-t5gemma]

Checks
  1. NeoBERT: load AutoModelForSequenceClassification (trust_remote_code), one
     forward+backward on a 16 x 256 batch in bf16, report peak VRAM.
  2. T5Gemma: for each candidate repo, load in bf16, run zero-shot generation on
     n gold_in rows from data/test.csv, print raw outputs and the fraction of
     outputs that contain the expected sentinel sequence.  A GatedRepoError is
     reported, not raised.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, load_entities, normalize  # noqa: E402

NEOBERT = "chandar-lab/NeoBERT"
T5GEMMA = ["google/t5gemma-2-270m-270m", "google/t5gemma-b-b-ul2"]
FALLBACK_T5 = "t5-base"


def vram() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB peak"


def test_neobert(test: pd.DataFrame) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"\n=== NeoBERT: {NEOBERT} ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(NEOBERT, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        NEOBERT, num_labels=3, trust_remote_code=True,
        problem_type="multi_label_classification", dtype=torch.bfloat16)
    from neobert_patch import use_eager_swiglu
    print("SwiGLU modules switched to eager op:", use_eager_swiglu(model))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).train()
    print(f"loaded in {time.time() - t0:.1f}s; params {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    texts = test.raw_text.head(16).tolist()
    batch = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
    labels = torch.tensor(test[["minor", "gender", "kinship"]].head(16).values, dtype=torch.bfloat16, device=dev)
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    out = model(**batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.logits.float(), labels.float())
    loss.backward()
    print(f"forward+backward OK on 16x{batch['input_ids'].shape[1]}: loss {loss.item():.3f}, {vram()}")
    del model, out, loss
    torch.cuda.empty_cache() if dev == "cuda" else None


def test_t5(repo: str, test: pd.DataFrame, n: int) -> bool:
    from huggingface_hub.errors import GatedRepoError
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"\n=== T5-style infilling: {repo} ===")
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=torch.bfloat16)
    except GatedRepoError as e:
        print(f"GATED: accept the licence at https://huggingface.co/{repo} ({str(e)[:80]})")
        return False
    except Exception as e:
        print(f"LOAD FAIL: {type(e).__name__}: {str(e)[:300]}")
        return False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M; sentinel ids:",
          tok.convert_tokens_to_ids(["<extra_id_0>", "<extra_id_1>"]))
    rows = test[test.num_entities > 0].head(n)
    ok = 0
    for r in rows.itertuples(index=False):
        enc = tok(r.gold_in, return_tensors="pt", truncation=True, max_length=384).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=48, num_beams=4)
        txt = tok.decode(gen[0], skip_special_tokens=False)
        txt = re.sub(r"</?s>|<pad>", "", txt).strip()
        n_sent = len(load_entities(r.entities_json))
        has = all(f"<extra_id_{i}>" in txt for i in range(n_sent))
        ok += has
        print(f"[{'ok ' if has else 'BAD'}] gold: {r.gold_tgt}\n      pred: {txt[:160]}")
    print(f"sentinel-format outputs: {ok}/{len(rows)}  ({vram()})")
    del model
    torch.cuda.empty_cache() if dev == "cuda" else None
    return ok / max(len(rows), 1) >= 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--skip-neobert", action="store_true")
    ap.add_argument("--skip-t5gemma", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="also try t5-base")
    args = ap.parse_args()
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8")
    if not args.skip_neobert:
        test_neobert(test)
    verdict = {}
    if not args.skip_t5gemma:
        for repo in T5GEMMA:
            verdict[repo] = test_t5(repo, test, args.n)
    if args.fallback:
        verdict[FALLBACK_T5] = test_t5(FALLBACK_T5, test, args.n)
    print("\nzero-shot sentinel infilling usable:", verdict)
    print("If False for the chosen model, the non-adaptive attacker is trained on random spans of negative rows (design doc, section 1).")


if __name__ == "__main__":
    main()
