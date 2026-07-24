"""
embeddings.py

Shared sentence-embedding backend for the benchmark pipeline.

Both build_datastore.py (embedding the 20k-row corpus once) and annotate.py
(embedding query sentences for kNN retrieval) must use the exact same model
and pooling strategy, or retrieval similarity scores are meaningless. This
module is the single place that logic lives.

Uses a SimCSE checkpoint (https://arxiv.org/abs/2104.08821) run through
sentence-transformers. SimCSE checkpoints on the HF Hub are bare encoder
weights with no sentence-transformers pooling config attached, so the
Transformer + Pooling modules are wired up manually here, using [CLS]-token
pooling per the SimCSE paper (not mean pooling).
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer, models

DEFAULT_SIMCSE_MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"


def load_encoder(model_name: str = DEFAULT_SIMCSE_MODEL, device: str | None = None) -> SentenceTransformer:
    """Build a SentenceTransformer wrapping a SimCSE checkpoint (CLS-token pooling)."""
    word_embedding_model = models.Transformer(model_name)
    if hasattr(word_embedding_model, "get_embedding_dimension"):
        dim = word_embedding_model.get_embedding_dimension()
    else:  # older sentence-transformers versions
        dim = word_embedding_model.get_word_embedding_dimension()
    try:
        pooling_model = models.Pooling(dim, pooling_mode="cls")
    except TypeError:  # older sentence-transformers versions
        pooling_model = models.Pooling(dim, pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)


def embed(encoder: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts to L2-normalized float32 embeddings (cosine sim == dot product)."""
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
    )
    return embeddings.astype(np.float32)
