"""FAISS vector index wrapper with metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd


class FaissSemanticIndex:
    """Stores and retrieves document vectors with cosine similarity via inner product."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, df: pd.DataFrame, embeddings: np.ndarray) -> None:
        if len(df) != len(embeddings):
            raise ValueError("Dataframe size and embeddings size must match.")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = df[["document_id", "text", "original_category"]].to_dict("records")

    def search_similar_documents(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build_index first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            item["index_id"] = int(idx)
            results.append(item)
        return results

    def save(self, directory: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty index.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "semantic.index"))
        (directory / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / "semantic.index"))
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
