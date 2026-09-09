# Knowledge Graph Builder - FastAPI REST API
# Author: Maharshi Soni | License: MIT

"""FastAPI application exposing endpoints for document ingestion,
graph queries, and statistics."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.graph_builder import GraphBuilder
from src.models import (
    BulkIngestRequest,
    DocumentInput,
    GraphStatsResponse,
    IngestResponse,
    NeighborRequest,
    NeighborResponse,
    PathResponse,
    ShortestPathRequest,
)
from src.synthetic_data import get_synthetic_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

builder = GraphBuilder()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-load synthetic data so the graph is non-empty on first visit."""
    docs = get_synthetic_documents()
    builder.ingest_documents(docs)
    logger.info("Startup: ingested %d synthetic documents.", len(docs))
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Knowledge Graph Builder API",
    description="Extract entities and relationships from text, build and query a knowledge graph.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(doc: DocumentInput) -> IngestResponse:
    """Ingest a single document and extract entities/relationships."""
    entities, relationships = builder.ingest_document(doc)
    return IngestResponse(
        documents_processed=1,
        entities_extracted=len(entities),
        relationships_extracted=len(relationships),
        total_graph_entities=builder.graph.entity_count,
        total_graph_relationships=builder.graph.relationship_count,
    )


@app.post("/ingest/bulk", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_bulk(request: BulkIngestRequest) -> IngestResponse:
    """Ingest multiple documents in a single request."""
    return builder.ingest_documents(request.documents)


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------

@app.post("/query/shortest-path", response_model=PathResponse, tags=["Queries"])
async def query_shortest_path(request: ShortestPathRequest) -> PathResponse:
    """Find the shortest path between two entities."""
    return builder.graph.shortest_path(request.source, request.target)


@app.post("/query/neighbors", response_model=NeighborResponse, tags=["Queries"])
async def query_neighbors(request: NeighborRequest) -> NeighborResponse:
    """Explore an entity's neighborhood up to N hops."""
    return builder.graph.neighbors(request.entity, request.depth)


@app.get("/query/components", tags=["Queries"])
async def query_components() -> Dict[str, Any]:
    """Return all connected components in the graph."""
    components = builder.graph.connected_components()
    return {
        "count": len(components),
        "components": components,
    }


@app.get("/query/pagerank", tags=["Queries"])
async def query_pagerank(top_n: int = 10) -> List[Dict[str, Any]]:
    """Return the top-N entities by PageRank score."""
    return builder.graph.pagerank(top_n)


# ---------------------------------------------------------------------------
# Statistics & info endpoints
# ---------------------------------------------------------------------------

@app.get("/stats", response_model=GraphStatsResponse, tags=["Statistics"])
async def graph_stats() -> GraphStatsResponse:
    """Return summary statistics about the knowledge graph."""
    return builder.graph.stats()


@app.get("/entities", tags=["Statistics"])
async def list_entities() -> List[Dict[str, Any]]:
    """List all entities in the graph."""
    return [
        {
            "name": e.name,
            "entity_type": e.entity_type.value,
            "mentions": e.mentions,
        }
        for e in builder.graph.get_all_entities()
    ]


@app.get("/relationships", tags=["Statistics"])
async def list_relationships() -> List[Dict[str, Any]]:
    """List all relationships in the graph."""
    return builder.graph.get_all_relationships()


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}
