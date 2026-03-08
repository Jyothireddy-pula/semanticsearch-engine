"""Fuzzy clustering utilities using Gaussian Mixture Models (soft clustering)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass
class ClusterSelectionResult:
    best_k: int
    scores: pd.DataFrame


class FuzzyClusterer:
    """Fits and serves GMM models with per-document membership probabilities."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: GaussianMixture | None = None
        self.membership: np.ndarray | None = None

    def choose_cluster_count(self, embeddings: np.ndarray, k_min: int = 5, k_max: int = 20) -> ClusterSelectionResult:
        rows: List[Dict[str, float]] = []
        candidate_models: Dict[int, GaussianMixture] = {}

        for k in range(k_min, k_max + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=self.random_state)
            labels = gmm.fit_predict(embeddings)
            rows.append(
                {
                    "k": float(k),
                    "bic": float(gmm.bic(embeddings)),
                    "aic": float(gmm.aic(embeddings)),
                    "silhouette": float(silhouette_score(embeddings, labels) if len(set(labels)) > 1 else -1.0),
                }
            )
            candidate_models[k] = gmm

        score_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        best_row = score_df.sort_values(["bic", "aic"], ascending=[True, True]).iloc[0]
        best_k = int(best_row["k"])

        self.model = candidate_models[best_k]
        self.membership = self.model.predict_proba(embeddings)
        return ClusterSelectionResult(best_k=best_k, scores=score_df)

    def fit(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        self.model = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=self.random_state)
        self.model.fit(embeddings)
        self.membership = self.model.predict_proba(embeddings)
        return self.membership

    def dominant_cluster(self, embedding: np.ndarray) -> Tuple[int, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Cluster model not fitted")
        probs = self.model.predict_proba(embedding.reshape(1, -1))[0]
        return int(np.argmax(probs)), probs

    def analyze_membership(self, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        if self.membership is None:
            raise RuntimeError("Membership is not available")

        out = df[["document_id", "text", "original_category"]].copy()
        out["dominant_cluster"] = self.membership.argmax(axis=1)
        out["dominant_confidence"] = self.membership.max(axis=1)
        out["membership_entropy"] = -(self.membership * np.log(self.membership + 1e-12)).sum(axis=1)

        reps = out.sort_values(["dominant_cluster", "dominant_confidence"], ascending=[True, False]).groupby("dominant_cluster").head(top_n)
        boundaries = out.sort_values("dominant_confidence", ascending=True).head(top_n)
        uncertain = out.sort_values("membership_entropy", ascending=False).head(top_n)

        return pd.concat(
            [reps.assign(role="representative"), boundaries.assign(role="boundary"), uncertain.assign(role="uncertain")],
            ignore_index=True,
        )

    def projection_dataframe(self, embeddings: np.ndarray, categories: list[str], document_ids: list[str]) -> pd.DataFrame:
        if self.membership is None:
            raise RuntimeError("Membership is not available")

        coords = PCA(n_components=2, random_state=self.random_state).fit_transform(embeddings)
        return pd.DataFrame(
            {
                "document_id": document_ids,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "dominant_cluster": self.membership.argmax(axis=1),
                "category": categories,
            }
        )

    def plot_clusters(self, embeddings: np.ndarray, output_path: str = "cluster_plot.png") -> str:
        if self.membership is None:
            raise RuntimeError("Membership is not available")
        coords = PCA(n_components=2, random_state=self.random_state).fit_transform(embeddings)
        labels = self.membership.argmax(axis=1)
        plt.figure(figsize=(10, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=8, alpha=0.7)
        plt.title("20 Newsgroups semantic clusters (PCA projection)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path

    def save(self, model_path: str | Path, membership_path: str | Path | None = None) -> None:
        if self.model is None:
            raise RuntimeError("No fitted model to save")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        if membership_path is not None and self.membership is not None:
            membership_path = Path(membership_path)
            membership_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(membership_path, self.membership)

    def load(self, model_path: str | Path, membership_path: str | Path | None = None) -> None:
        self.model = joblib.load(model_path)
        if membership_path is not None and Path(membership_path).exists():
            self.membership = np.load(membership_path)
