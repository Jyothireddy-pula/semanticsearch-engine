"""Application entrypoint for semantic search API."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from api.routes import container, router
from search.semantic_search import build_artifacts, load_service_from_artifacts


def create_app() -> FastAPI:
    app = FastAPI(title="20 Newsgroups Semantic Search", version="2.0.0")
    app.include_router(router)

    @app.on_event("startup")
    def startup_event() -> None:
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
        dataset_root = os.getenv("DATASET_ROOT", "./data/20_newsgroups")
        auto_bootstrap = os.getenv("AUTO_BOOTSTRAP", "false").lower() == "true"

        if not (artifacts_dir / "manifest.json").exists():
            if not auto_bootstrap:
                raise RuntimeError(
                    "Artifacts not found. Run bootstrap.py first or set AUTO_BOOTSTRAP=true with DATASET_ROOT configured."
                )
            build_artifacts(dataset_root=dataset_root, artifacts_dir=artifacts_dir)

        container.service = load_service_from_artifacts(artifacts_dir)

    @app.get("/health")
    def health_check() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
