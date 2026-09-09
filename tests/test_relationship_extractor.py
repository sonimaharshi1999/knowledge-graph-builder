# Knowledge Graph Builder - Relationship Extractor Tests
# Author: Maharshi Soni | License: MIT

"""Tests for relationship extraction using verb patterns and co-occurrence."""

from __future__ import annotations

from src.models import Entity, EntityType, RelationshipType
from src.relationship_extractor import RelationshipExtractor


class TestRelationshipExtractor:
    """Test suite for RelationshipExtractor."""

    def test_works_at_pattern(self, relationship_extractor: RelationshipExtractor) -> None:
        """Verb pattern 'works at' should produce a WORKS_AT relationship."""
        text = "Alice Chen works at NovaTech in their San Francisco office."
        entities = [
            Entity(name="Alice Chen", entity_type=EntityType.PERSON),
            Entity(name="NovaTech", entity_type=EntityType.ORGANIZATION),
        ]
        rels = relationship_extractor.extract(text, entities)
        rel_types = [r.relation_type for r in rels]
        assert RelationshipType.WORKS_AT in rel_types

    def test_co_occurrence_fallback(self, relationship_extractor: RelationshipExtractor) -> None:
        """Entities in the same sentence without a verb pattern get CO_OCCURS."""
        text = "NovaTech and DataFlow Labs both serve enterprise customers."
        entities = [
            Entity(name="NovaTech", entity_type=EntityType.ORGANIZATION),
            Entity(name="DataFlow Labs", entity_type=EntityType.ORGANIZATION),
        ]
        rels = relationship_extractor.extract(text, entities)
        assert len(rels) >= 1
        assert any(r.relation_type == RelationshipType.CO_OCCURS for r in rels)

    def test_single_entity_yields_no_relationships(
        self, relationship_extractor: RelationshipExtractor
    ) -> None:
        """A single entity cannot form a relationship."""
        text = "NovaTech is a great company."
        entities = [Entity(name="NovaTech", entity_type=EntityType.ORGANIZATION)]
        rels = relationship_extractor.extract(text, entities)
        assert rels == []
