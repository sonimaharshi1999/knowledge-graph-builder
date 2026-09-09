# Knowledge Graph Builder - Data Models
# Author: Maharshi Soni | License: MIT

"""Pydantic models for entities, relationships, graph queries, and API responses."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Supported named-entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    TECHNOLOGY = "TECHNOLOGY"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    UNKNOWN = "UNKNOWN"


class RelationshipType(str, Enum):
    """Edge types in the knowledge graph."""
    WORKS_AT = "WORKS_AT"
    LOCATED_IN = "LOCATED_IN"
    USES = "USES"
    DEVELOPS = "DEVELOPS"
    PARTNERS_WITH = "PARTNERS_WITH"
    MANAGES = "MANAGES"
    CO_OCCURS = "CO_OCCURS"
    FOUNDED = "FOUNDED"
    ACQUIRED = "ACQUIRED"
    RELATED_TO = "RELATED_TO"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A named entity extracted from text."""
    name: str = Field(..., min_length=1, description="Canonical entity name")
    entity_type: EntityType = Field(..., description="Type classification")
    mentions: int = Field(default=1, ge=1, description="Number of mentions in corpus")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        return hash((self.name.lower(), self.entity_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.name.lower() == other.name.lower() and self.entity_type == other.entity_type


class Relationship(BaseModel):
    """A directed relationship between two entities."""
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation_type: RelationshipType = Field(..., description="Type of relationship")
    weight: float = Field(default=1.0, ge=0.0, description="Edge weight / confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class DocumentInput(BaseModel):
    """A single document submitted for ingestion."""
    text: str = Field(..., min_length=1, description="Raw text content")
    doc_id: Optional[str] = Field(default=None, description="Optional document identifier")


class BulkIngestRequest(BaseModel):
    """Batch of documents for bulk ingestion."""
    documents: List[DocumentInput] = Field(..., min_length=1)


class ShortestPathRequest(BaseModel):
    """Request to find shortest path between two entities."""
    source: str
    target: str


class NeighborRequest(BaseModel):
    """Request to explore an entity's neighborhood."""
    entity: str
    depth: int = Field(default=1, ge=1, le=5)


class GraphStatsResponse(BaseModel):
    """Summary statistics for the knowledge graph."""
    total_entities: int
    total_relationships: int
    entity_type_counts: Dict[str, int]
    relationship_type_counts: Dict[str, int]
    connected_components: int
    density: float
    avg_degree: float
    top_entities_by_degree: List[Dict[str, Any]]
    top_entities_by_pagerank: List[Dict[str, Any]]


class PathResponse(BaseModel):
    """Response containing a path between entities."""
    source: str
    target: str
    path: List[str]
    length: int
    exists: bool


class NeighborResponse(BaseModel):
    """Response containing an entity's neighborhood."""
    entity: str
    depth: int
    neighbors: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class IngestResponse(BaseModel):
    """Response after ingesting document(s)."""
    documents_processed: int
    entities_extracted: int
    relationships_extracted: int
    total_graph_entities: int
    total_graph_relationships: int
