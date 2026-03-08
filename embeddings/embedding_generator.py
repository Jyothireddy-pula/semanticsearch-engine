"""Sentence transformer embedding generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Wraps a sentence-transformers model for batched inference."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: Iterable[str], batch_size: int = 64, show_progress_bar: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text], show_progress_bar=False)[0]

    @staticmethod
    def save_embeddings(path: str | Path, embeddings: np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    @staticmethod
    def load_embeddings(path: str | Path) -> np.ndarray:
        return np.load(path)
