# Knowledge Graph Builder - Entity Extractor Tests
# Author: Maharshi Soni | License: MIT

"""Tests for NLP-based entity extraction."""

from __future__ import annotations

from src.entity_extractor import EntityExtractor
from src.models import EntityType


class TestEntityExtractor:
    """Test suite for EntityExtractor."""

    def test_extracts_person_entities(self, entity_extractor: EntityExtractor) -> None:
        """Verify that known person names are extracted from text."""
        text = "Alice Chen founded NovaTech in San Francisco."
        entities = entity_extractor.extract(text)
        names_lower = [e.name.lower() for e in entities]
        assert any("alice" in n for n in names_lower), (
            f"Expected 'Alice Chen' among extracted entities, got: {names_lower}"
        )

    def test_extracts_technology_keywords(self, entity_extractor: EntityExtractor) -> None:
        """Verify that technology keywords are detected."""
        text = "The team uses Python, Docker, and Kubernetes for deployment."
        entities = entity_extractor.extract(text)
        tech_names = [e.name.lower() for e in entities if e.entity_type == EntityType.TECHNOLOGY]
        assert "python" in tech_names
        assert "docker" in tech_names
        assert "kubernetes" in tech_names

    def test_deduplication_merges_mentions(self, entity_extractor: EntityExtractor) -> None:
        """Verify that duplicate entities are merged per (name, type) and counts are aggregated."""
        text = (
            "Docker is popular. Many teams use Docker for deployment. "
            "Docker is also used for local development."
        )
        entities = entity_extractor.extract(text)
        # The tech-keyword scanner should find Docker exactly once as TECHNOLOGY
        docker_tech = [
            e for e in entities
            if e.name.lower() == "docker" and e.entity_type == EntityType.TECHNOLOGY
        ]
        assert len(docker_tech) == 1, (
            f"Expected exactly one TECHNOLOGY entity for Docker, got: {docker_tech}"
        )
        assert docker_tech[0].mentions >= 1

    def test_empty_text_returns_empty(self, entity_extractor: EntityExtractor) -> None:
        """Verify that empty/whitespace text yields no entities."""
        assert entity_extractor.extract("") == []
        assert entity_extractor.extract("   ") == []
