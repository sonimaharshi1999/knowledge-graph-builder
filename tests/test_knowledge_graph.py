# Knowledge Graph Builder - Knowledge Graph Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the NetworkX-backed knowledge graph: CRUD, queries, and stats."""

from __future__ import annotations

from typing import List

from src.knowledge_graph import KnowledgeGraph
from src.models import Entity, EntityType, Relationship, RelationshipType


class TestKnowledgeGraph:
    """Test suite for KnowledgeGraph."""

    def test_add_entity_and_count(self, empty_graph: KnowledgeGraph) -> None:
        """Adding entities should increase the node count."""
        assert empty_graph.entity_count == 0
        empty_graph.add_entity(Entity(name="Alice", entity_type=EntityType.PERSON))
        assert empty_graph.entity_count == 1
        empty_graph.add_entity(Entity(name="Bob", entity_type=EntityType.PERSON))
        assert empty_graph.entity_count == 2

    def test_add_duplicate_entity_merges(self, empty_graph: KnowledgeGraph) -> None:
        """Adding the same entity twice should merge mentions, not duplicate."""
        empty_graph.add_entity(Entity(name="Alice", entity_type=EntityType.PERSON, mentions=2))
        empty_graph.add_entity(Entity(name="Alice", entity_type=EntityType.PERSON, mentions=3))
        assert empty_graph.entity_count == 1
        entity = empty_graph.get_entity("Alice")
        assert entity is not None
        assert entity.mentions == 5

    def test_add_relationship_creates_edge(self, empty_graph: KnowledgeGraph) -> None:
        """Adding a relationship should create an edge and auto-create missing nodes."""
        rel = Relationship(
            source="Alice", target="NovaTech", relation_type=RelationshipType.WORKS_AT
        )
        empty_graph.add_relationship(rel)
        assert empty_graph.entity_count == 2
        assert empty_graph.relationship_count == 1

    def test_shortest_path_exists(self, populated_graph: KnowledgeGraph) -> None:
        """Shortest path between connected entities should be found."""
        result = populated_graph.shortest_path("Alice Chen", "Python")
        assert result.exists is True
        assert result.length >= 1
        assert result.path[0] == "Alice Chen"
        assert result.path[-1] == "Python"

    def test_shortest_path_no_path(self, empty_graph: KnowledgeGraph) -> None:
        """Query between disconnected nodes should report no path."""
        empty_graph.add_entity(Entity(name="A", entity_type=EntityType.PERSON))
        empty_graph.add_entity(Entity(name="B", entity_type=EntityType.PERSON))
        result = empty_graph.shortest_path("A", "B")
        assert result.exists is False
        assert result.path == []

    def test_neighbors_returns_adjacent(self, populated_graph: KnowledgeGraph) -> None:
        """Neighbor query should return directly connected entities."""
        result = populated_graph.neighbors("NovaTech", depth=1)
        neighbor_names = [n["name"] for n in result.neighbors]
        assert len(neighbor_names) >= 1

    def test_connected_components(self, populated_graph: KnowledgeGraph) -> None:
        """The sample graph should form at least one connected component."""
        components = populated_graph.connected_components()
        assert len(components) >= 1
        # All 5 sample entities are connected
        assert len(components[0]) == 5

    def test_pagerank_returns_scores(self, populated_graph: KnowledgeGraph) -> None:
        """PageRank should return scored entities."""
        pr = populated_graph.pagerank(top_n=3)
        assert len(pr) >= 1
        assert "entity" in pr[0]
        assert "score" in pr[0]
        assert pr[0]["score"] > 0

    def test_stats_response_fields(self, populated_graph: KnowledgeGraph) -> None:
        """Stats should contain all expected summary fields."""
        stats = populated_graph.stats()
        assert stats.total_entities == 5
        assert stats.total_relationships == 5
        assert stats.connected_components >= 1
        assert stats.density > 0
        assert len(stats.top_entities_by_degree) >= 1


class TestGraphBuilderIntegration:
    """Integration tests for the full GraphBuilder pipeline."""

    def test_ingest_synthetic_documents(self, graph_builder, synthetic_docs) -> None:
        """Ingesting the full synthetic corpus should produce a non-trivial graph."""
        result = graph_builder.ingest_documents(synthetic_docs)
        assert result.documents_processed == 10
        assert result.entities_extracted > 10
        assert result.relationships_extracted > 0
        assert result.total_graph_entities > 5
