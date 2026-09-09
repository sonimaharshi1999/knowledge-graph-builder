# Knowledge Graph Builder - Graph Builder Orchestrator
# Author: Maharshi Soni | License: MIT

"""High-level orchestrator that ties together entity extraction,
relationship extraction, and the knowledge graph store."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from src.entity_extractor import EntityExtractor
from src.knowledge_graph import KnowledgeGraph
from src.models import DocumentInput, Entity, IngestResponse, Relationship
from src.relationship_extractor import RelationshipExtractor

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Orchestrates the full pipeline: text -> entities -> relationships -> graph."""

    def __init__(self) -> None:
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.graph = KnowledgeGraph()

    def ingest_document(self, doc: DocumentInput) -> Tuple[List[Entity], List[Relationship]]:
        """Process a single document and add results to the graph.

        Returns the entities and relationships extracted from this document.
        """
        entities = self.entity_extractor.extract(doc.text)
        relationships = self.relationship_extractor.extract(doc.text, entities)

        self.graph.add_entities(entities)
        self.graph.add_relationships(relationships)

        logger.info(
            "Ingested doc=%s  entities=%d  relationships=%d",
            doc.doc_id or "<unnamed>",
            len(entities),
            len(relationships),
        )
        return entities, relationships

    def ingest_documents(self, documents: List[DocumentInput]) -> IngestResponse:
        """Process a batch of documents and return aggregate stats."""
        total_entities = 0
        total_relationships = 0

        for doc in documents:
            entities, relationships = self.ingest_document(doc)
            total_entities += len(entities)
            total_relationships += len(relationships)

        return IngestResponse(
            documents_processed=len(documents),
            entities_extracted=total_entities,
            relationships_extracted=total_relationships,
            total_graph_entities=self.graph.entity_count,
            total_graph_relationships=self.graph.relationship_count,
        )

    def get_stats(self) -> Dict:
        """Return graph statistics as a dict."""
        return self.graph.stats().model_dump()
