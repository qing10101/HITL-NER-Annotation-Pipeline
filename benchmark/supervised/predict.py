"""predict.py

Unified inference: runs any of the four trained models over a gold-format
CSV and writes a predictions CSV (row_id, predicted_entities_json) with
char offsets on raw_text — the format benchmark/llm/evaluate.py scores.

Usage:
    python benchmark/supervised/predict.py --model spacy \
        --input benchmark/supervised/data/splits/test.csv \
        --output benchmark/supervised/predictions/spacy.csv

    python benchmark/llm/evaluate.py \
        --gold benchmark/supervised/data/splits/test.csv \
        --pred spacy=benchmark/supervised/predictions/spacy.csv \
        --pred bilstm=benchmark/supervised/predictions/bilstm.csv \
        --pred spanmarker=benchmark/supervised/predictions/spanmarker.csv \
        --pred gliner=benchmark/supervised/predictions/gliner.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent

DEFAULT_MODEL_DIRS = {
    "spacy": HERE / "models" / "spacy" / "model-best",
    "bilstm": HERE / "models" / "bilstm_crf",
    "spanmarker": HERE / "models" / "spanmarker" / "model-best",
    "gliner": HERE / "models" / "gliner" / "model-best",
}


def predict_spacy(model_dir: str, texts: list[str], batch_size: int,
                  **_) -> list[list[dict]]:
    import spacy
    nlp = spacy.load(model_dir)
    preds = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        preds.append([
            {"label": ent.label_, "text": ent.text,
             "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ])
    return preds


def predict_bilstm(model_dir: str, texts: list[str], batch_size: int,
                   **_) -> list[list[dict]]:
    import torch
    import train_bilstm_crf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = train_bilstm_crf.load_model(model_dir, device)
    return train_bilstm_crf.predict_texts(bundle, texts, device, batch_size)


def predict_spanmarker(model_dir: str, texts: list[str], batch_size: int,
                       **_) -> list[list[dict]]:
    import torch
    from span_marker import SpanMarkerModel
    model = SpanMarkerModel.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model = model.cuda()
    results = model.predict(texts, batch_size=batch_size)
    preds = []
    for text, ents in zip(texts, results):
        preds.append([
            {"label": e["label"], "text": e["span"],
             "start": e["char_start_index"], "end": e["char_end_index"]}
            for e in ents
        ])
    return preds


def predict_gliner(model_dir: str, texts: list[str], batch_size: int,
                   threshold: float = 0.5) -> list[list[dict]]:
    import torch
    from gliner import GLiNER
    model = GLiNER.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
    phrases = list(common.PHRASE_TO_LABEL)
    preds = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        batch_ents = model.batch_predict_entities(chunk, phrases,
                                                  threshold=threshold)
        for ents in batch_ents:
            preds.append([
                {"label": common.PHRASE_TO_LABEL[e["label"]], "text": e["text"],
                 "start": e["start"], "end": e["end"]}
                for e in ents
            ])
    return preds


PREDICTORS = {
    "spacy": predict_spacy,
    "bilstm": predict_bilstm,
    "spanmarker": predict_spanmarker,
    "gliner": predict_gliner,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a trained model over a CSV.")
    ap.add_argument("--model", required=True, choices=sorted(PREDICTORS))
    ap.add_argument("--model-dir", default=None,
                    help="Defaults to benchmark/supervised/models/<model>/…")
    ap.add_argument("--input", default=str(HERE / "data" / "splits" / "test.csv"))
    ap.add_argument("--output", default=None,
                    help="Defaults to benchmark/supervised/predictions/<model>.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Confidence threshold (gliner only).")
    args = ap.parse_args()

    model_dir = args.model_dir or str(DEFAULT_MODEL_DIRS[args.model])
    output = args.output or str(HERE / "predictions" / f"{args.model}.csv")

    examples = common.load_examples(args.input)
    texts = [ex.text for ex in examples]
    print(f"Predicting {len(texts)} rows with {args.model} ({model_dir})")

    preds = PREDICTORS[args.model](
        model_dir, texts, args.batch_size, threshold=args.threshold,
    )

    common.write_predictions_csv(
        output, [(ex.row_id, ents) for ex, ents in zip(examples, preds)]
    )
    n_ents = sum(len(p) for p in preds)
    print(f"Wrote {len(preds)} rows ({n_ents} entities) to {output}")


if __name__ == "__main__":
    main()
