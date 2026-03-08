"""FastAPI route definitions for semantic search system."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from search.semantic_search import SemanticSearchService


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    dominant_cluster: int
    documents: list[dict]


class ServiceContainer:
    service: SemanticSearchService | None = None


container = ServiceContainer()
router = APIRouter()


def get_service() -> SemanticSearchService:
    if container.service is None:
        raise HTTPException(status_code=503, detail="Service is not initialized")
    return container.service


@router.post("/query", response_model=QueryResponse)
def query_semantic_search(
    request: QueryRequest,
    service: SemanticSearchService = Depends(get_service)
) -> dict:
    return service.query(query_text=request.query, top_k=request.top_k)


@router.get("/cache/stats")
def cache_stats(service: SemanticSearchService = Depends(get_service)) -> dict:
    return service.cache.stats()


@router.post("/cache/threshold")
def set_cache_threshold(
    request: CacheThresholdRequest,
    service: SemanticSearchService = Depends(get_service),
) -> dict:
    service.cache.set_threshold(request.similarity_threshold)
    return service.cache.stats()


@router.delete("/cache")
def clear_cache(service: SemanticSearchService = Depends(get_service)) -> dict:
    service.cache.clear()
    return {"message": "Cache cleared", **service.cache.stats()}
