"""Semantic search orchestration across embeddings, vector index, clustering, and cache."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd # type: ignore

from cache.semantic_cache import SemanticCache
from clustering.fuzzy_cluster import FuzzyClusterer
from data.dataset_loader import NewsgroupsDatasetLoader
from embeddings.embedding_generator import EmbeddingGenerator
from preprocessing.text_cleaner import TextCleaner
from vector_store.faiss_index import FaissSemanticIndex


class SemanticSearchService:
    """Online semantic retrieval service."""
    """Online semantic retrieval service that coordinates embedding, search, clustering and caching."""

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        index: FaissSemanticIndex,
        clusterer: FuzzyClusterer,
        cache: SemanticCache,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.clusterer = clusterer
        self.cache = cache

    @staticmethod
    def _summarize(query: str, results: List[Dict[str, Any]], dominant_cluster: int) -> str:
        """Create a readable summary of search results."""
        if not results:
            return f"No strong semantic matches found for '{query}'."

        category_counts = Counter(r["original_category"] for r in results)

        top_categories = ", ".join(
            f"{cat} ({count})" for cat, count in category_counts.most_common(3)
        )

        snippets = " | ".join(r["text"][:140].replace("\n", " ") for r in results[:3])

        return (
            f"Cluster {dominant_cluster} appears to represent the main intent. "
            f"Top categories: {top_categories}. "
            f"Example snippets: {snippets}"
        )

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Run semantic search for a query."""

        query_embedding = self.embedder.encode_query(query_text)

        dominant_cluster, _ = self.clusterer.dominant_cluster(query_embedding)

        cached = self.cache.get(query_embedding, dominant_cluster)

        if cached is not None:
            return cached

        documents = self.index.search_similar_documents(query_embedding, top_k=top_k)

        summary = self._summarize(query_text, documents, dominant_cluster)

        response: Dict[str, Any] = {
            "query": query_text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": summary,
            "dominant_cluster": dominant_cluster,
            "documents": documents,
        }

        self.cache.add(query_text, query_embedding, response, dominant_cluster)

        return response


def build_artifacts(
    dataset_root: str | Path,
    artifacts_dir: str | Path,
    min_words: int = 10,
    k_min: int = 8,
    k_max: int = 20,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Offline pipeline that prepares all artifacts required for the API."""

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_df = NewsgroupsDatasetLoader(dataset_root).load()

    clean_df = TextCleaner(min_words=min_words).clean_dataframe(raw_df)

    clean_df.to_parquet(artifacts_dir / "clean_corpus.parquet", index=False)

    embedder = EmbeddingGenerator(model_name=model_name)

    embeddings = embedder.encode_texts(
        clean_df["text"].tolist(),
        show_progress_bar=True,
    )

    embedder.save_embeddings(artifacts_dir / "embeddings.npy", embeddings)

    index = FaissSemanticIndex()
    index.build_index(clean_df, embeddings)
    index.save(artifacts_dir / "index")

    clusterer = FuzzyClusterer()

    selection = clusterer.choose_cluster_count(
        embeddings,
        k_min=k_min,
        k_max=k_max,
    )

    analysis = clusterer.analyze_membership(clean_df, top_n=5)

    analysis.to_parquet(
        artifacts_dir / "cluster_analysis.parquet",
        index=False,
    )

    projection = clusterer.projection_dataframe(
        embeddings,
        categories=clean_df["original_category"].tolist(),
        document_ids=clean_df["document_id"].tolist(),
    )

    projection.to_parquet(
        artifacts_dir / "cluster_projection.parquet",
        index=False,
    )

    selection.scores.to_parquet(
        artifacts_dir / "cluster_selection.parquet",
        index=False,
    )

    clusterer.save(
        artifacts_dir / "gmm.joblib",
        artifacts_dir / "membership.npy",
    )

    manifest = {
        "dataset_root": str(dataset_root),
        "model_name": model_name,
        "documents_loaded": int(len(raw_df)),
        "documents_after_cleaning": int(len(clean_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "best_cluster_count": int(selection.best_k),
    }

    (artifacts_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


def load_service_from_artifacts(
    artifacts_dir: str | Path,
    cache_threshold: float = 0.85,
) -> SemanticSearchService:
    """Load all saved artifacts and initialize the semantic search service."""

    artifacts = Path(artifacts_dir)

    manifest = json.loads(
        (artifacts / "manifest.json").read_text(encoding="utf-8")
    )

    embedder = EmbeddingGenerator(model_name=manifest["model_name"])

    index = FaissSemanticIndex()
    index.load(artifacts / "index")

    clusterer = FuzzyClusterer()

    membership_path = artifacts / "membership.npy"

    clusterer.load(
        artifacts / "gmm.joblib",
        membership_path if membership_path.exists() else None,
    )

    cache = SemanticCache(similarity_threshold=cache_threshold)

    return SemanticSearchService(
        embedder=embedder,
        index=index,
        clusterer=clusterer,
        cache=cache,
    )
