# Knowledge Graph Builder - API Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the FastAPI REST API using httpx AsyncClient."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import app, builder
from src.synthetic_data import get_synthetic_documents


@pytest.fixture(scope="module", autouse=True)
def _seed_graph() -> None:
    """Seed the API's shared graph builder with synthetic data once per module.

    The FastAPI lifespan does this at server startup, but httpx's
    ASGITransport does not invoke lifespan events, so we replicate the
    seeding here for the test suite.
    """
    if builder.graph.entity_count == 0:
        docs = get_synthetic_documents()
        builder.ingest_documents(docs)


@pytest.fixture
async def client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client) -> None:
    """Health check should return 200 with status healthy."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_stats_endpoint(client) -> None:
    """Stats endpoint should return graph summary with expected fields."""
    resp = await client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_entities" in data
    assert "total_relationships" in data
    assert data["total_entities"] > 0


@pytest.mark.asyncio
async def test_entities_endpoint(client) -> None:
    """Entities endpoint should return a list of entity dicts."""
    resp = await client.get("/entities")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]
    assert "entity_type" in data[0]


@pytest.mark.asyncio
async def test_ingest_single_document(client) -> None:
    """Ingesting a single document should return extraction counts."""
    payload = {
        "text": "John Smith works at Acme Corp in Chicago. Acme Corp uses Java.",
        "doc_id": "test-001",
    }
    resp = await client.post("/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_processed"] == 1
    assert data["entities_extracted"] >= 0


@pytest.mark.asyncio
async def test_shortest_path_endpoint(client) -> None:
    """Shortest path query should return a valid response structure."""
    payload = {"source": "NovaTech", "target": "Python"}
    resp = await client.post("/query/shortest-path", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "exists" in data
    assert "path" in data


@pytest.mark.asyncio
async def test_neighbors_endpoint(client) -> None:
    """Neighbors query should return neighborhood data."""
    payload = {"entity": "NovaTech", "depth": 1}
    resp = await client.post("/query/neighbors", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "neighbors" in data
    assert "edges" in data


@pytest.mark.asyncio
async def test_pagerank_endpoint(client) -> None:
    """PageRank endpoint should return scored entities."""
    resp = await client.get("/query/pagerank?top_n=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_components_endpoint(client) -> None:
    """Components endpoint should return component list."""
    resp = await client.get("/query/components")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "components" in data
