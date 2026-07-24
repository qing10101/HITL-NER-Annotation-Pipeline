# Supervised NER Benchmark

Trains four supervised NER models on `benchmark/gold_standard_merged.csv`
(20k rows, 5 labels: FAM_KIN, MINOR_AGE, GEN_NOUN, GEN_PHYS, MINOR_EDU) and
scores them with the same exact-span evaluator as the LLM benchmark
(`benchmark/llm/evaluate.py`), so the LLM and supervised results are directly comparable.

| Trainer | Model | Approach |
|---|---|---|
| `train_spacy.py` | spaCy `ner` (tok2vec, or roberta-base with `--trf`) | transition-based token tagging |
| `train_bilstm_crf.py` | word emb + char-CNN + BiLSTM + CRF (PyTorch, no pretraining) | BIO sequence labeling |
| `train_spanmarker.py` | SpanMarker + `microsoft/deberta-v3-base` | span enumeration + classification |
| `train_gliner.py` | fine-tuned `urchade/gliner_medium-v2.1` | span ↔ label-phrase matching |

## Setup

```bash
conda activate ner-train
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-supervised.txt
```

## Workflow

```bash
cd benchmark/supervised

# 1. One shared grouped+stratified 80/10/10 split, plus per-model formats.
#    --neg-ratio 2.0 caps train at 2 empty rows per entity-bearing row.
python prepare_data.py --neg-ratio 2.0

# 2. Train (any order; each early-stops/selects on dev F1)
python train_spacy.py                 # add --trf --gpu 0 for the transformer variant
python train_bilstm_crf.py
python train_gliner.py
python train_spanmarker.py

# 3. Predict on the held-out test split
python predict.py --model spacy
python predict.py --model bilstm
python predict.py --model spanmarker
python predict.py --model gliner --threshold 0.5

# 4. Score all four side by side
python ../llm/evaluate.py --gold data/splits/test.csv \
    --pred spacy=predictions/spacy.csv \
    --pred bilstm=predictions/bilstm.csv \
    --pred spanmarker=predictions/spanmarker.csv \
    --pred gliner=predictions/gliner.csv
```

## Design notes

- **Split integrity**: rows are grouped by the product-id prefix of `row_id`
  (near-duplicate reviews of one product never straddle splits) and groups
  are stratified by their rarest label, so MINOR_EDU (~97 mentions) and
  GEN_PHYS (~164) appear in every split. Same split feeds all four models.
- **Offsets are the contract**: every model's predictions are converted back
  to character offsets on `raw_text` before scoring; tokenization lives in
  `common.tokenize`, which records offsets so BIO round-trips are exact.
- **GLiNER label phrases**: GLiNER learns to match spans against
  natural-language label descriptions (`common.GLINER_LABEL_PHRASES`). Train
  and predict must use identical phrases; predict.py maps them back to codes.
- **Caveat**: per-label F1 for MINOR_EDU and GEN_PHYS will be noisy — the
  10% test slice holds only ~10 and ~16 gold entities respectively.

## Tests

```bash
python -m unittest tests.test_ml_common -v   # from repo root; stdlib-only
```
