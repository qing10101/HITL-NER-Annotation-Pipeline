"""train_bilstm_crf.py

Self-contained PyTorch BiLSTM-CRF sequence tagger trained on the BIO files
from prepare_data.py. No pretrained weights — word embeddings are learned
from scratch, with a char-CNN so unseen words ("3yo", "kindergartner") still
get a signature from their spelling. A CRF output layer (pytorch-crf) scores
whole tag sequences, ruling out incoherent transitions like O -> I-KINSHIP.

Early-stops on dev micro span-F1 (exact label+offset match, same criterion
as the final evaluate.py scoring). The checkpoint bundles the vocabularies
and hyperparameters so predict.py can restore it with load_model().

Usage:
    python benchmark/supervised/train_bilstm_crf.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchcrf import CRF

import common

PAD, UNK = "<pad>", "<unk>"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_bio_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_vocabs(rows: list[dict], min_freq: int) -> tuple[dict, dict]:
    word_counts = Counter(t.lower() for r in rows for t in r["tokens"])
    char_counts = Counter(c for r in rows for t in r["tokens"] for c in t)
    word_vocab = {PAD: 0, UNK: 1}
    for w, n in word_counts.most_common():
        if n >= min_freq:
            word_vocab[w] = len(word_vocab)
    char_vocab = {PAD: 0, UNK: 1}
    for c, _ in char_counts.most_common():
        char_vocab[c] = len(char_vocab)
    return word_vocab, char_vocab


def encode_row(row: dict, word_vocab: dict, char_vocab: dict, max_word_len: int) -> dict:
    word_ids = [word_vocab.get(t.lower(), word_vocab[UNK]) for t in row["tokens"]]
    char_ids = [
        [char_vocab.get(c, char_vocab[UNK]) for c in t[:max_word_len]]
        for t in row["tokens"]
    ]
    tag_ids = [common.BIO_TAG_TO_ID[t] for t in row["tags"]]
    return {"words": word_ids, "chars": char_ids, "tags": tag_ids, "row": row}


def make_batches(encoded: list[dict], batch_size: int, rng: random.Random | None) -> list[list[dict]]:
    """Length-bucketed batches: sort by length, cut, then shuffle batch order."""
    order = sorted(encoded, key=lambda r: len(r["words"]))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(batch: list[dict], device: torch.device) -> tuple[torch.Tensor, ...]:
    max_len = max(len(r["words"]) for r in batch)
    max_chars = max((len(cs) for r in batch for cs in r["chars"]), default=1)
    words = torch.zeros(len(batch), max_len, dtype=torch.long)
    chars = torch.zeros(len(batch), max_len, max_chars, dtype=torch.long)
    tags = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, r in enumerate(batch):
        n = len(r["words"])
        words[i, :n] = torch.tensor(r["words"])
        tags[i, :n] = torch.tensor(r["tags"])
        mask[i, :n] = True
        for j, cs in enumerate(r["chars"]):
            if cs:
                chars[i, j, :len(cs)] = torch.tensor(cs)
    return words.to(device), chars.to(device), tags.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiLstmCrf(nn.Module):
    def __init__(self, n_words: int, n_chars: int, n_tags: int, hp: dict):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, hp["word_dim"], padding_idx=0)
        self.char_emb = nn.Embedding(n_chars, hp["char_dim"], padding_idx=0)
        self.char_cnn = nn.Conv1d(hp["char_dim"], hp["char_filters"],
                                  kernel_size=3, padding=1)
        self.dropout = nn.Dropout(hp["dropout"])
        self.lstm = nn.LSTM(
            hp["word_dim"] + hp["char_filters"], hp["hidden"] // 2,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.proj = nn.Linear(hp["hidden"], n_tags)
        self.crf = CRF(n_tags, batch_first=True)

    def _features(self, words: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        B, T, C = chars.shape
        ch = self.char_emb(chars.view(B * T, C)).transpose(1, 2)  # (B*T, dim, C)
        ch = torch.relu(self.char_cnn(ch)).max(dim=2).values       # (B*T, filters)
        ch = ch.view(B, T, -1)
        x = torch.cat([self.word_emb(words), ch], dim=-1)
        x, _ = self.lstm(self.dropout(x))
        return self.proj(self.dropout(x))

    def loss(self, words, chars, tags, mask) -> torch.Tensor:
        emissions = self._features(words, chars)
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, words, chars, mask) -> list[list[int]]:
        return self.crf.decode(self._features(words, chars), mask=mask)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def predict_rows(model: BiLstmCrf, encoded: list[dict], batch_size: int,
                 device: torch.device) -> list[list[dict]]:
    """Decode rows (in their given order) back to char-offset entity dicts."""
    model.eval()
    preds: list[list[dict]] = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            words, chars, _, mask = collate(batch, device)
            for r, tag_ids in zip(batch, model.decode(words, chars, mask)):
                row = r["row"]
                tokens = list(zip(row["tokens"], row["starts"], row["ends"]))
                tags = [common.BIO_TAGS[t] for t in tag_ids]
                preds.append(common.bio_to_spans(row["text"], tokens, tags))
    return preds


def evaluate(model: BiLstmCrf, encoded: list[dict], batch_size: int,
             device: torch.device) -> tuple[float, float, float]:
    gold = [
        common.bio_to_spans(
            r["row"]["text"],
            list(zip(r["row"]["tokens"], r["row"]["starts"], r["row"]["ends"])),
            r["row"]["tags"],
        )
        for r in encoded
    ]
    pred = predict_rows(model, encoded, batch_size, device)
    return common.span_micro_f1(gold, pred)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Train a BiLSTM-CRF tagger.")
    ap.add_argument("--data-dir", default=str(here / "data"))
    ap.add_argument("--out-dir", default=str(here / "models" / "bilstm_crf"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--word-dim", type=int, default=100)
    ap.add_argument("--char-dim", type=int, default=30)
    ap.add_argument("--char-filters", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-word-len", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    train_rows = read_bio_jsonl(data_dir / "bio" / "train.jsonl")
    dev_rows = read_bio_jsonl(data_dir / "bio" / "dev.jsonl")
    word_vocab, char_vocab = build_vocabs(train_rows, args.min_freq)
    print(f"Vocab: {len(word_vocab)} words, {len(char_vocab)} chars, "
          f"{len(common.BIO_TAGS)} tags")

    train_enc = [encode_row(r, word_vocab, char_vocab, args.max_word_len)
                 for r in train_rows]
    dev_enc = [encode_row(r, word_vocab, char_vocab, args.max_word_len)
               for r in dev_rows]

    hp = {k: getattr(args, k) for k in
          ("word_dim", "char_dim", "char_filters", "hidden", "dropout")}
    model = BiLstmCrf(len(word_vocab), len(char_vocab), len(common.BIO_TAGS), hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"

    best_f1, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = make_batches(train_enc, args.batch_size, rng)
        for batch in batches:
            words, chars, tags, mask = collate(batch, device)
            loss = model.loss(words, chars, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        prec, rec, f1 = evaluate(model, dev_enc, args.batch_size, device)
        marker = ""
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "word_vocab": word_vocab,
                "char_vocab": char_vocab,
                "hparams": hp,
                "max_word_len": args.max_word_len,
            }, ckpt_path)
            marker = "  *saved*"
        print(f"epoch {epoch:02d}  loss={total_loss / len(batches):.4f}  "
              f"dev P={prec:.3f} R={rec:.3f} F1={f1:.3f}{marker}")
        if epoch - best_epoch >= args.patience:
            print(f"Early stop: no dev-F1 gain in {args.patience} epochs.")
            break

    print(f"Best dev F1={best_f1:.3f} (epoch {best_epoch}). Checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Inference API used by predict.py
# ---------------------------------------------------------------------------

def load_model(model_dir: str | Path, device: torch.device) -> dict:
    ckpt = torch.load(Path(model_dir) / "model.pt", map_location=device,
                      weights_only=False)
    model = BiLstmCrf(len(ckpt["word_vocab"]), len(ckpt["char_vocab"]),
                      len(common.BIO_TAGS), ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ckpt["model"] = model
    return ckpt


def predict_texts(bundle: dict, texts: list[str], device: torch.device,
                  batch_size: int = 64) -> list[list[dict]]:
    encoded = []
    for text in texts:
        tokens = common.tokenize(text)
        row = {
            "text": text,
            "tokens": [t for t, _, _ in tokens],
            "starts": [s for _, s, _ in tokens],
            "ends": [e for _, _, e in tokens],
            "tags": ["O"] * len(tokens),
        }
        encoded.append(encode_row(row, bundle["word_vocab"], bundle["char_vocab"],
                                  bundle["max_word_len"]))
    # predict_rows preserves input order; rows with no tokens would crash the
    # collate step, so map them straight to empty predictions instead.
    non_empty = [e for e in encoded if e["words"]]
    preds_iter = iter(predict_rows(bundle["model"], non_empty, batch_size, device))
    return [next(preds_iter) if e["words"] else [] for e in encoded]


if __name__ == "__main__":
    main()
