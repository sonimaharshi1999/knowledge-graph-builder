# Knowledge Graph Builder - Test Fixtures
# Author: Maharshi Soni | License: MIT

"""Shared pytest fixtures for entity extraction, graph building, and API testing."""

from __future__ import annotations

from typing import List

import pytest

from src.entity_extractor import EntityExtractor
from src.graph_builder import GraphBuilder
from src.knowledge_graph import KnowledgeGraph
from src.models import DocumentInput, Entity, EntityType, Relationship, RelationshipType
from src.relationship_extractor import RelationshipExtractor
from src.synthetic_data import get_synthetic_documents


@pytest.fixture
def entity_extractor() -> EntityExtractor:
    return EntityExtractor()


@pytest.fixture
def relationship_extractor() -> RelationshipExtractor:
    return RelationshipExtractor()


@pytest.fixture
def empty_graph() -> KnowledgeGraph:
    return KnowledgeGraph()


@pytest.fixture
def sample_entities() -> List[Entity]:
    return [
        Entity(name="Alice Chen", entity_type=EntityType.PERSON, mentions=3),
        Entity(name="NovaTech", entity_type=EntityType.ORGANIZATION, mentions=5),
        Entity(name="San Francisco", entity_type=EntityType.LOCATION, mentions=2),
        Entity(name="Python", entity_type=EntityType.TECHNOLOGY, mentions=4),
        Entity(name="Bob Martinez", entity_type=EntityType.PERSON, mentions=2),
    ]


@pytest.fixture
def sample_relationships() -> List[Relationship]:
    return [
        Relationship(source="Alice Chen", target="NovaTech", relation_type=RelationshipType.WORKS_AT),
        Relationship(source="NovaTech", target="San Francisco", relation_type=RelationshipType.LOCATED_IN),
        Relationship(source="NovaTech", target="Python", relation_type=RelationshipType.USES),
        Relationship(source="Bob Martinez", target="NovaTech", relation_type=RelationshipType.WORKS_AT),
        Relationship(source="Alice Chen", target="Bob Martinez", relation_type=RelationshipType.MANAGES),
    ]


@pytest.fixture
def populated_graph(
    empty_graph: KnowledgeGraph,
    sample_entities: List[Entity],
    sample_relationships: List[Relationship],
) -> KnowledgeGraph:
    empty_graph.add_entities(sample_entities)
    empty_graph.add_relationships(sample_relationships)
    return empty_graph


@pytest.fixture
def graph_builder() -> GraphBuilder:
    return GraphBuilder()


@pytest.fixture
def synthetic_docs() -> List[DocumentInput]:
    return get_synthetic_documents()
