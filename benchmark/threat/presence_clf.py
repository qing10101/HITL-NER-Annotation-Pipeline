"""Scheme B: dimension-presence attacker (minor / gender / kinship).

    python benchmark/threat/presence_clf.py --model tfidf   --train-on raw      # A_raw, no GPU needed
    python benchmark/threat/presence_clf.py --model tfidf   --train-on gold_del # A_adapt
    python benchmark/threat/presence_clf.py --model neobert --train-on raw --seed 42   # .venv-threat
    python benchmark/threat/presence_clf.py --model neobert --train-on gold_del --seed 42

Train text column = raw_text (non-adaptive attacker) or gold_del (adaptive).
Every run is scored on the test set under three conditions: raw, gold_del, rand_del.
Outputs in benchmark/threat/results/presence_<model>_<train-on>_s<seed>/:
    preds_<cond>.csv   row_id, minor, gender, kinship (true), p_minor, p_gender, p_kinship
    metrics.json       per-condition, per-label AUROC / F1@dev-thr / mean p on positives /
                       share of p in [0.4,0.6] / Brier / attack success (p>=0.8 and correct)
    cues.json          (tfidf only) top positive word features per label
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR  # noqa: E402

LABELS = ["minor", "gender", "kinship"]
CONDS = {"raw": "raw_text", "gold_del": "gold_del", "rand_del": "rand_del"}
RESULTS = Path(__file__).resolve().parent / "results"
HF_NAMES = {"neobert": "chandar-lab/NeoBERT", "roberta": "roberta-base",
            "distilroberta": "distilroberta-base"}


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def tune_thresholds(y_dev: np.ndarray, p_dev: np.ndarray) -> list[float]:
    from sklearn.metrics import f1_score
    thr = []
    for j in range(y_dev.shape[1]):
        best, bt = -1, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            f = f1_score(y_dev[:, j], p_dev[:, j] >= t, zero_division=0)
            if f > best:
                best, bt = f, float(t)
        thr.append(bt)
    return thr


def score(y: np.ndarray, p: np.ndarray, thr: list[float]) -> dict:
    from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
    out = {}
    for j, lab in enumerate(LABELS):
        yj, pj = y[:, j], p[:, j]
        pos = yj == 1
        pred = (pj >= thr[j]).astype(int)
        out[lab] = {
            "n_pos": int(pos.sum()),
            "auroc": float(roc_auc_score(yj, pj)) if 0 < pos.sum() < len(yj) else None,
            "f1": float(f1_score(yj, pred, zero_division=0)),
            "precision": float(((pred == 1) & pos).sum() / max((pred == 1).sum(), 1)),
            "recall": float(((pred == 1) & pos).sum() / max(pos.sum(), 1)),
            "mean_p_pos": float(pj[pos].mean()) if pos.any() else None,
            "mean_p_neg": float(pj[~pos].mean()) if (~pos).any() else None,
            "share_uncertain": float(((pj >= 0.4) & (pj <= 0.6)).mean()),
            "brier": float(brier_score_loss(yj, pj)),
            "attack_success_pos": float(((pj >= 0.8) & pos).sum() / max(pos.sum(), 1)),
            "false_alarm_neg": float(((pj >= 0.8) & ~pos).sum() / max((~pos).sum(), 1)),
            "threshold": thr[j],
        }
    aur = [v["auroc"] for v in out.values() if v["auroc"] is not None]
    out["macro"] = {"auroc": float(np.mean(aur)) if aur else None,
                    "f1": float(np.mean([v["f1"] for v in out.values() if isinstance(v, dict) and "f1" in v]))}
    return out


# ----------------------------------------------------------------------------
# TF-IDF + logistic regression
# ----------------------------------------------------------------------------
def run_tfidf(train, dev, test, train_col, out_dir, seed):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    Xtr = hstack([wv.fit_transform(train[train_col].fillna("")), cv.fit_transform(train[train_col].fillna(""))]).tocsr()

    def feats(texts):
        return hstack([wv.transform(texts.fillna("")), cv.transform(texts.fillna(""))]).tocsr()

    ytr = train[LABELS].values
    clfs, cues = [], {}
    n_word = len(wv.vocabulary_)
    vocab = np.array(wv.get_feature_names_out())
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000, random_state=seed)
        clf.fit(Xtr, ytr[:, j])
        clfs.append(clf)
        w = clf.coef_[0][:n_word]
        top = np.argsort(-w)[:40]
        cues[lab] = [(vocab[i], round(float(w[i]), 3)) for i in top]

    def predict(texts):
        X = feats(texts)
        return np.column_stack([c.predict_proba(X)[:, 1] for c in clfs])

    p_dev = predict(dev[train_col])
    thr = tune_thresholds(dev[LABELS].values, p_dev)
    metrics = {"dev": score(dev[LABELS].values, p_dev, thr)}
    for cond, col in CONDS.items():
        p = predict(test[col])
        metrics[cond] = score(test[LABELS].values, p, thr)
        save_preds(test, p, out_dir / f"preds_{cond}.csv")
    (out_dir / "cues.json").write_text(json.dumps(cues, indent=1, ensure_ascii=False), encoding="utf-8")
    return metrics


# ----------------------------------------------------------------------------
# HF encoder (NeoBERT / RoBERTa)
# ----------------------------------------------------------------------------
def run_hf(train, dev, test, train_col, out_dir, seed, model_key, args):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(seed)
    name = HF_NAMES[model_key]
    trust = model_key == "neobert"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(LABELS), trust_remote_code=trust,
        problem_type="multi_label_classification")
    if trust:
        from neobert_patch import enable_layer_checkpointing, use_eager_swiglu
        print("SwiGLU eager:", use_eager_swiglu(model),
              "| layers checkpointed:", enable_layer_checkpointing(model) if not args.no_ckpt else 0)

    def to_ds(df, col):
        d = Dataset.from_pandas(pd.DataFrame({"text": df[col].fillna("").values,
                                              "labels": df[LABELS].values.astype(np.float32).tolist()}))
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                     batched=True, remove_columns=["text"])

    ds_tr, ds_dev = to_ds(train, train_col), to_ds(dev, train_col)
    pos = train[LABELS].values.sum(0)
    pos_weight = torch.tensor((len(train) - pos) / np.maximum(pos, 1), dtype=torch.float32).clamp(max=20.0)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out.logits.float(), labels.float(), pos_weight=pos_weight.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def compute_metrics(ev):
        from sklearn.metrics import roc_auc_score
        logits, y = ev
        p = 1 / (1 + np.exp(-logits))
        au = [roc_auc_score(y[:, j], p[:, j]) for j in range(len(LABELS)) if 0 < y[:, j].sum() < len(y)]
        return {"macro_auc": float(np.mean(au))}

    import math
    steps_per_epoch = math.ceil(len(ds_tr) / (args.bs * args.accum))
    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpt"), seed=seed,
        learning_rate=args.lr, per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum, per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs, warmup_steps=max(1, int(0.06 * steps_per_epoch * args.epochs)),
        weight_decay=0.01,
        bf16=torch.cuda.is_available(), eval_strategy="epoch", save_strategy="no",
        logging_steps=50, report_to=[], dataloader_num_workers=0)
    # save_strategy="no": NeoBERT's config carries max_length=4096, which transformers 5
    # refuses to write via save_pretrained; weights are persisted as a state_dict below.
    # The final epoch is used (no best-checkpoint reload).
    trainer = WTrainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_dev,
                       data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()
    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    if torch.cuda.is_available():
        print(f"peak VRAM during training: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    def predict(df, col):
        logits = trainer.predict(to_ds(df, col)).predictions
        return 1 / (1 + np.exp(-logits))

    p_dev = predict(dev, train_col)
    thr = tune_thresholds(dev[LABELS].values, p_dev)
    metrics = {"dev": score(dev[LABELS].values, p_dev, thr)}
    for cond, col in CONDS.items():
        p = predict(test, col)
        metrics[cond] = score(test[LABELS].values, p, thr)
        save_preds(test, p, out_dir / f"preds_{cond}.csv")
    (out_dir / "model").mkdir(exist_ok=True)
    torch.save(trainer.model.state_dict(), out_dir / "model" / "state_dict.pt")
    return metrics


def save_preds(test, p, path):
    df = test[["row_id"] + LABELS].copy()
    for j, lab in enumerate(LABELS):
        df[f"p_{lab}"] = p[:, j]
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf", "neobert", "roberta", "distilroberta"], default="tfidf")
    ap.add_argument("--train-on", choices=["raw", "gold_del"], default="raw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--no-ckpt", action="store_true", help="disable per-layer activation checkpointing (NeoBERT)")
    ap.add_argument("--limit", type=int, default=None, help="debug: subsample train rows")
    ap.add_argument("--dev-limit", type=int, default=None, help="debug: subsample dev rows")
    args = ap.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8")
    dev = pd.read_csv(DATA_DIR / "dev.csv", encoding="utf-8")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8")
    if args.limit:
        train = train.sample(args.limit, random_state=args.seed)
    if args.dev_limit:
        dev = dev.sample(args.dev_limit, random_state=args.seed)
    train_col = "raw_text" if args.train_on == "raw" else "gold_del"
    out_dir = RESULTS / f"presence_{args.model}_{args.train_on}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "tfidf":
        metrics = run_tfidf(train, dev, test, train_col, out_dir, args.seed)
    else:
        metrics = run_hf(train, dev, test, train_col, out_dir, args.seed, args.model, args)
    metrics["config"] = vars(args) | {"train_rows": len(train)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    print(f"\n{args.model} trained on {args.train_on}  (seed {args.seed})")
    print(f"{'cond':9} {'label':8} {'AUROC':>6} {'F1':>6} {'P':>6} {'R':>6} {'p+':>6} {'p-':>6} {'succ':>6}")
    for cond in CONDS:
        for lab in LABELS:
            m = metrics[cond][lab]
            print(f"{cond:9} {lab:8} {m['auroc'] or 0:6.3f} {m['f1']:6.3f} {m['precision']:6.3f} "
                  f"{m['recall']:6.3f} {m['mean_p_pos'] or 0:6.3f} {m['mean_p_neg'] or 0:6.3f} {m['attack_success_pos']:6.3f}")
        print(f"{cond:9} {'macro':8} {metrics[cond]['macro']['auroc'] or 0:6.3f} {metrics[cond]['macro']['f1']:6.3f}")


if __name__ == "__main__":
    main()
