"""Custom semantic cache implementation (in-memory, no external backend)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    result: Dict[str, Any]
    dominant_cluster: int
    timestamp: datetime


class SemanticCache:
    """In-memory semantic cache with cluster-aware lookup."""

    def __init__(self, similarity_threshold: float = 0.85, max_entries: int = 1000) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        self.entries: List[CacheEntry] = []
        self.cluster_to_indices: Dict[int, List[int]] = {}

        self.hit_count: int = 0
        self.miss_count: int = 0

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
        return float(np.dot(vec_a, vec_b) / denom)

    def get(self, query_embedding: np.ndarray, dominant_cluster: int) -> Optional[Dict[str, Any]]:
        """Return cached result if similarity threshold is met."""

        candidate_indices = self.cluster_to_indices.get(dominant_cluster, [])

        best_score: float = -1.0
        best_entry: Optional[CacheEntry] = None

        for idx in candidate_indices:
            entry = self.entries[idx]

            score = self._cosine_similarity(query_embedding, entry.query_embedding)

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            self.hit_count += 1

            cached_result = dict(best_entry.result)
            cached_result.update(
                {
                    "cache_hit": True,
                    "matched_query": best_entry.query_text,
                    "similarity_score": round(best_score, 4),
                }
            )

            return cached_result

        self.miss_count += 1
        return None

    def add(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: Dict[str, Any],
        dominant_cluster: int,
    ) -> None:
        """Add a new entry to the cache."""

        if len(self.entries) >= self.max_entries:
            self._evict_oldest()

        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding.copy(),
            result=result,
            dominant_cluster=dominant_cluster,
            timestamp=datetime.now(timezone.utc),
        )

        self.entries.append(entry)

        idx = len(self.entries) - 1
        self.cluster_to_indices.setdefault(dominant_cluster, []).append(idx)

    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry when capacity is reached."""
        if not self.entries:
            return

        oldest_index = min(range(len(self.entries)), key=lambda i: self.entries[i].timestamp)
        oldest_entry = self.entries[oldest_index]

        self.entries.pop(oldest_index)

        if oldest_entry.dominant_cluster in self.cluster_to_indices:
            self.cluster_to_indices[oldest_entry.dominant_cluster].remove(oldest_index)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries.clear()
        self.cluster_to_indices.clear()
        self.hit_count = 0
        self.miss_count = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = len(self.entries)
        lookups = self.hit_count + self.miss_count

        hit_rate = (self.hit_count / lookups) if lookups > 0 else 0.0

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
        }
