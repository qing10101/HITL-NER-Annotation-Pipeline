"""Scheme A: span reconstruction attacker (T5-style sentinel infilling).

    # non-adaptive: has never seen a privacy span; trained to infill random
    # spans of negative (span-free) reviews only
    python benchmark/threat/recon_t5.py --model t5gemma2 --mode nonadaptive --seed 42
    # adaptive: trained to infill the gold spans of the training pool
    python benchmark/threat/recon_t5.py --model t5gemma2 --mode adaptive --seed 42
    # score an existing run again without training
    python benchmark/threat/recon_t5.py --model t5gemma2 --mode adaptive --seed 42 --eval-only

Input/target format (common.sentinel_pair):
    in : Bought this for my <extra_id_0> <extra_id_1>.  He likes it ...
    tgt: <extra_id_0> 1 year old <extra_id_1> nephew <extra_id_2>

Test conditions (same model, different inputs):
    gold : all gold spans masked (test rows with >=1 span)
    loo  : one gold span masked at a time, others visible (test_loo.csv)
    rand : equal-count random word spans masked (test rows with >=1 span)

Outputs in benchmark/threat/results/recon_<model>_<mode>_s<seed>/:
    preds_<cond>.csv  one row per masked span: row_id, span_idx, label, gold,
                      pred1..pred5, exact1, exact5, cat1, gold_prob
    metrics.json      per condition x label: exact@1, exact@5, category@1, mean gold prob
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, LABELS, load_entities, normalize  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MODELS = {"t5gemma2": "google/t5gemma-2-270m-270m",
          "t5gemma-ul2": "google/t5gemma-b-b-ul2",
          "t5-base": "t5-base"}
N_SENT = 32
SENT_RE = re.compile(r"<extra_id_(\d+)>")
UNUSED_RE = re.compile(r"<unused(\d+)>")
# Gemma vocabularies have no <extra_id_k>, but ship ~99 atomic <unusedN> tokens.
# When the tokenizer lacks <extra_id_0> we map <extra_id_k> <-> <unusedk> at the
# tokenizer boundary; the CSVs and all scoring keep the <extra_id_k> spelling.
GEMMA_SENTINELS = False


def to_model(text: str) -> str:
    return SENT_RE.sub(lambda m: f"<unused{m.group(1)}>", text) if GEMMA_SENTINELS else text


def from_model(text: str) -> str:
    return UNUSED_RE.sub(lambda m: f"<extra_id_{m.group(1)}>", text) if GEMMA_SENTINELS else text


def sentinel_token(i: int) -> str:
    return f"<unused{i}>" if GEMMA_SENTINELS else f"<extra_id_{i}>"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def build_train_sets(train: pd.DataFrame, dev: pd.DataFrame, mode: str, seed: int,
                     n_neg_adaptive: int = 800):
    """Return (train_pairs, dev_pairs) as DataFrames with columns src, tgt."""
    rng = np.random.RandomState(seed)
    pos_tr, neg_tr = train[train.num_entities > 0], train[train.num_entities == 0]
    pos_dv, neg_dv = dev[dev.num_entities > 0], dev[dev.num_entities == 0]
    if mode == "adaptive":
        tr = pd.concat([pos_tr[["gold_in", "gold_tgt"]].rename(columns={"gold_in": "src", "gold_tgt": "tgt"}),
                        neg_tr.sample(min(n_neg_adaptive, len(neg_tr)), random_state=rng)[["rand_in", "rand_tgt"]]
                        .rename(columns={"rand_in": "src", "rand_tgt": "tgt"})])
        dv = pd.concat([pos_dv[["gold_in", "gold_tgt"]].rename(columns={"gold_in": "src", "gold_tgt": "tgt"}),
                        neg_dv.sample(min(100, len(neg_dv)), random_state=rng)[["rand_in", "rand_tgt"]]
                        .rename(columns={"rand_in": "src", "rand_tgt": "tgt"})])
    else:  # nonadaptive: random spans of negative rows only, matched in size to the adaptive set
        n = min(len(pos_tr) + n_neg_adaptive, len(neg_tr))
        tr = neg_tr.sample(n, random_state=rng)[["rand_in", "rand_tgt"]].rename(columns={"rand_in": "src", "rand_tgt": "tgt"})
        dv = neg_dv.sample(min(len(pos_dv) + 100, len(neg_dv)), random_state=rng)[["rand_in", "rand_tgt"]] \
            .rename(columns={"rand_in": "src", "rand_tgt": "tgt"})
    return tr.sample(frac=1, random_state=rng).reset_index(drop=True), dv.reset_index(drop=True)


def test_sets(test: pd.DataFrame, loo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pos = test[test.num_entities > 0]
    gold = pd.DataFrame({"row_id": pos.row_id, "src": pos.gold_in, "tgt": pos.gold_tgt,
                         "spans": pos.entities_json})
    rand = pd.DataFrame({"row_id": pos.row_id, "src": pos.rand_in, "tgt": pos.rand_tgt,
                         "spans": pos.rand_json})
    loo_df = pd.DataFrame({"row_id": loo.row_id, "src": loo.loo_in, "tgt": loo.loo_tgt,
                           "spans": [json.dumps([{"label": l, "text": t, "start": 0, "end": 0}])
                                     for l, t in zip(loo.label, loo.span_text)],
                           "span_idx": loo.span_idx})
    return {"gold": gold, "loo": loo_df, "rand": rand}


# ----------------------------------------------------------------------------
# model / tokenizer
# ----------------------------------------------------------------------------
def load_model_and_tok(name: str, path: str | None = None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    src = path or name
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSeq2SeqLM.from_pretrained(src, dtype=torch.bfloat16)
    global GEMMA_SENTINELS
    unk = tok.unk_token_id
    has_extra = tok.convert_tokens_to_ids("<extra_id_0>") not in (None, unk)
    has_unused = all(tok.convert_tokens_to_ids(f"<unused{i}>") not in (None, unk) for i in range(N_SENT))
    if has_extra:
        GEMMA_SENTINELS = False
    elif has_unused:
        GEMMA_SENTINELS = True
        print(f"tokenizer has no <extra_id_k>; using <unused0>..<unused{N_SENT - 1}> as sentinels")
    else:
        raise RuntimeError("tokenizer has neither <extra_id_k> nor <unusedN> tokens")
    # freeze the (unused) vision tower of T5Gemma 2
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model {name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, {trainable / 1e6:.0f}M trainable")
    return model, tok


def make_encoder(tok, max_in: int, max_out: int):
    eos = tok.eos_token_id

    def enc(batch):
        x = tok([to_model(t) for t in batch["src"]], truncation=True, max_length=max_in)
        y = tok([to_model(t) for t in batch["tgt"]], truncation=True, max_length=max_out - 1,
                add_special_tokens=False)["input_ids"]
        x["labels"] = [ids + [eos] for ids in y]
        return x
    return enc


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def split_sentinels(text: str, k: int) -> list[str]:
    """'<extra_id_0> a b <extra_id_1> c <extra_id_2>' -> ['a b', 'c'] (first k fills)."""
    text = re.sub(r"<pad>|</s>|<bos>|<eos>|<s>", "", text)
    parts = SENT_RE.split(text)
    # parts = [prefix, idx0, fill0, idx1, fill1, ...]
    fills = {}
    for i in range(1, len(parts) - 1, 2):
        idx = int(parts[i])
        if idx not in fills:
            fills[idx] = parts[i + 1].strip()
    return [fills.get(i, "") for i in range(k)]


@torch.no_grad()
def gold_span_probs(model, tok, src: str, tgt: str, k: int, dev: str, max_in: int, max_out: int) -> list[float]:
    """Teacher-forced mean token log-prob per span, exponentiated."""
    x = tok(to_model(src), return_tensors="pt", truncation=True, max_length=max_in).to(dev)
    y = tok(to_model(tgt), add_special_tokens=False, truncation=True, max_length=max_out - 1)["input_ids"] + [tok.eos_token_id]
    labels = torch.tensor([y], device=dev)
    logits = model(**x, labels=labels).logits.float()
    lp = torch.log_softmax(logits, -1)[0, torch.arange(len(y)), labels[0]].tolist()
    sent_ids = {tok.convert_tokens_to_ids(sentinel_token(i)): i for i in range(N_SENT)}
    cur, acc = None, {i: [] for i in range(k)}
    for t, l in zip(y, lp):
        if t in sent_ids:
            cur = sent_ids[t]
            continue
        if cur is not None and cur in acc:
            acc[cur].append(l)
    return [math.exp(np.mean(v)) if v else float("nan") for v in acc.values()]


def run_condition(model, tok, df: pd.DataFrame, lexicon: dict, dev: str, args) -> pd.DataFrame:
    rows = []
    model.eval()
    for start in range(0, len(df), args.gen_bs):
        chunk = df.iloc[start:start + args.gen_bs]
        x = tok([to_model(t) for t in chunk.src.tolist()], return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_in).to(dev)
        with torch.no_grad():
            gen = model.generate(**x, max_new_tokens=args.max_out, num_beams=args.beams,
                                 num_return_sequences=args.beams, do_sample=False)
        texts = [from_model(t) for t in tok.batch_decode(gen, skip_special_tokens=False)]
        for bi, r in enumerate(chunk.itertuples(index=False)):
            spans = json.loads(r.spans)
            k = len(spans)
            cands = [split_sentinels(texts[bi * args.beams + j], k) for j in range(args.beams)]
            probs = gold_span_probs(model, tok, r.src, r.tgt, k, dev, args.max_in, args.max_out)
            for si, sp in enumerate(spans):
                gold_n = normalize(sp["text"])
                preds = [c[si] for c in cands]
                preds_n = [normalize(p) for p in preds]
                lab = sp["label"]
                lex = lexicon.get(lab, {})
                any_lex = set().union(*[set(v) for v in lexicon.values()])
                rows.append({
                    "row_id": r.row_id, "span_idx": getattr(r, "span_idx", si), "label": lab,
                    "gold": sp["text"], "gold_norm": gold_n,
                    **{f"pred{j + 1}": preds[j] for j in range(args.beams)},
                    "exact1": int(preds_n[0] == gold_n),
                    "exact5": int(gold_n in preds_n),
                    "cat1": int(preds_n[0] in lex) if lab in LABELS else None,
                    "anycat1": int(preds_n[0] in any_lex),
                    "gold_prob": probs[si] if si < len(probs) else float("nan"),
                })
    return pd.DataFrame(rows)


def summarize(pred: pd.DataFrame) -> dict:
    out = {}
    groups = [("all", pred)] + [(l, pred[pred.label == l]) for l in sorted(pred.label.unique())]
    for name, g in groups:
        if len(g) == 0:
            continue
        out[name] = {"n": int(len(g)),
                     "exact1": float(g.exact1.mean()), "exact5": float(g.exact5.mean()),
                     "cat1": float(g.cat1.mean()) if g.cat1.notna().any() else None,
                     "anycat1": float(g.anycat1.mean()),
                     "gold_prob": float(g.gold_prob.mean(skipna=True))}
    return out


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="t5gemma2")
    ap.add_argument("--mode", choices=["nonadaptive", "adaptive"], default="adaptive")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-in", type=int, default=384)
    ap.add_argument("--max-out", type=int, default=64)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--gen-bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap train pairs")
    ap.add_argument("--test-limit", type=int, default=None, help="debug: cap rows per test condition")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--fp32", action="store_true",
                    help="keep trainable (text) weights in fp32 master precision; compute still bf16 autocast")
    ap.add_argument("--optim", choices=["adafactor", "adamw_torch"], default="adafactor")
    ap.add_argument("--tag", default="", help="suffix for the output directory, e.g. fp32")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = RESULTS / f"recon_{args.model}_{args.mode}{tag}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8")
    devdf = pd.read_csv(DATA_DIR / "dev.csv", encoding="utf-8")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8")
    loo = pd.read_csv(DATA_DIR / "test_loo.csv", encoding="utf-8")
    lexicon = json.loads((DATA_DIR / "lexicon.json").read_text(encoding="utf-8"))

    model_dir = out_dir / "model"
    if args.eval_only and (model_dir / "state_dict.pt").exists():
        # transformers 5.22 mis-maps T5Gemma2 keys on save_pretrained/from_pretrained,
        # so weights are persisted as a plain state_dict against the live module names.
        model, tok = load_model_and_tok(MODELS[args.model])
        sd = torch.load(model_dir / "state_dict.pt", map_location="cpu")
        res = model.load_state_dict(sd, strict=False)
        print(f"state_dict loaded: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
        assert not res.unexpected_keys and all("vision_tower" in k or "multi_modal" in k for k in res.missing_keys), res
    else:
        from datasets import Dataset
        from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                                  Seq2SeqTrainingArguments, set_seed)
        set_seed(args.seed)
        model, tok = load_model_and_tok(MODELS[args.model])
        if args.fp32:  # bf16-stored weights + Adafactor at lr~1e-4 lose sub-ulp updates (learns the
            n_cast = 0  # output prior, not the context mapping); keep trainable weights in fp32
            for p_ in model.parameters():
                if p_.requires_grad:
                    p_.data = p_.data.float()
                    n_cast += p_.numel()
            print(f"fp32 master weights for {n_cast / 1e6:.0f}M trainable params (bf16 autocast for compute)")
        tr, dv = build_train_sets(train, devdf, args.mode, args.seed)
        if args.limit:
            tr, dv = tr.head(args.limit), dv.head(max(50, args.limit // 10))
        print(f"train pairs {len(tr)}, dev pairs {len(dv)}")
        enc = make_encoder(tok, args.max_in, args.max_out)
        ds_tr = Dataset.from_pandas(tr).map(enc, batched=True, remove_columns=["src", "tgt"])
        ds_dv = Dataset.from_pandas(dv).map(enc, batched=True, remove_columns=["src", "tgt"])
        steps_per_epoch = math.ceil(len(ds_tr) / (args.bs * args.accum))
        warmup = max(1, int(0.05 * steps_per_epoch * args.epochs))
        targs = Seq2SeqTrainingArguments(
            output_dir=str(out_dir / "ckpt"), seed=args.seed,
            learning_rate=args.lr, optim=args.optim, lr_scheduler_type="linear", warmup_steps=warmup,
            per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
            per_device_eval_batch_size=8, num_train_epochs=args.epochs,
            bf16=(dev == "cuda"), gradient_checkpointing=True,
            eval_strategy="epoch", save_strategy="no",
            logging_steps=50, report_to=[], dataloader_num_workers=0, predict_with_generate=False)
        trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_dv,
                                 data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100))
        trainer.train()
        model_dir.mkdir(parents=True, exist_ok=True)
        model = trainer.model
        torch.save({k: v for k, v in model.state_dict().items()
                    if "vision_tower" not in k and "multi_modal" not in k}, model_dir / "state_dict.pt")
        (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
        model.gradient_checkpointing_disable()
        model.config.use_cache = True

    model.to(dev)
    metrics = {}
    for cond, df in test_sets(test, loo).items():
        if args.test_limit:
            df = df.head(args.test_limit)
        pred = run_condition(model, tok, df, lexicon, dev, args)
        pred.to_csv(out_dir / f"preds_{cond}.csv", index=False, encoding="utf-8")
        metrics[cond] = summarize(pred)
        print(f"\n[{cond}] n={len(pred)}")
        for name, m in metrics[cond].items():
            print(f"  {name:10} n={m['n']:4d} exact@1={m['exact1']:.3f} exact@5={m['exact5']:.3f} "
                  f"cat@1={(m['cat1'] if m['cat1'] is not None else float('nan')):.3f} "
                  f"anycat@1={m['anycat1']:.3f} gold_p={m['gold_prob']:.3f}")
    metrics["config"] = vars(args)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
