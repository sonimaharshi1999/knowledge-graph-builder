# Knowledge Graph Builder - Entity Extractor
# Author: Maharshi Soni | License: MIT

"""Extract named entities from unstructured text using NLTK's NE chunker
and supplementary heuristics for technology / product names."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

import nltk

from src.models import Entity, EntityType
from src.nlp_utils import ensure_nltk_resources, split_sentences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK NE label -> our EntityType mapping
# ---------------------------------------------------------------------------

_NLTK_LABEL_MAP: Dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "ORGANIZATION": EntityType.ORGANIZATION,
    "GPE": EntityType.LOCATION,
    "GSP": EntityType.LOCATION,
    "LOCATION": EntityType.LOCATION,
    "FACILITY": EntityType.LOCATION,
}

# Technology / product keywords detected via heuristic patterns
_TECH_KEYWORDS: Set[str] = {
    "python", "java", "javascript", "typescript", "rust", "go", "c++",
    "kubernetes", "docker", "aws", "azure", "gcp", "terraform",
    "react", "angular", "vue", "node.js", "fastapi", "django", "flask",
    "postgresql", "mongodb", "redis", "kafka", "elasticsearch",
    "graphql", "rest", "grpc", "microservices", "serverless",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "ci/cd", "jenkins", "github actions", "gitlab ci",
    "linux", "windows", "macos",
}

# Pattern for likely proper nouns that NLTK may miss (CamelCase, acronyms)
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extracts named entities from raw text using NLTK + heuristics."""

    def __init__(self) -> None:
        ensure_nltk_resources()

    def extract(self, text: str) -> List[Entity]:
        """Return deduplicated entities found in *text*."""
        raw_entities: List[Tuple[str, EntityType]] = []

        # 1. NLTK NE chunker
        raw_entities.extend(self._nltk_extract(text))

        # 2. Technology keyword scan
        raw_entities.extend(self._tech_keyword_scan(text))

        # 3. Proper-noun heuristic (backup)
        raw_entities.extend(self._proper_noun_heuristic(text))

        # Deduplicate and count mentions
        return self._deduplicate(raw_entities)

    # ------------------------------------------------------------------
    # Internal extraction strategies
    # ------------------------------------------------------------------

    def _nltk_extract(self, text: str) -> List[Tuple[str, EntityType]]:
        """Use NLTK's ne_chunk pipeline."""
        results: List[Tuple[str, EntityType]] = []
        for sentence in split_sentences(text):
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            tree = nltk.ne_chunk(tagged, binary=False)
            for subtree in tree:
                if hasattr(subtree, "label"):
                    entity_name = " ".join(word for word, _tag in subtree.leaves())
                    label = subtree.label()
                    entity_type = _NLTK_LABEL_MAP.get(label, EntityType.UNKNOWN)
                    if len(entity_name) >= 2:
                        results.append((entity_name, entity_type))
        return results

    def _tech_keyword_scan(self, text: str) -> List[Tuple[str, EntityType]]:
        """Scan for known technology keywords (case-insensitive)."""
        results: List[Tuple[str, EntityType]] = []
        text_lower = text.lower()
        for keyword in _TECH_KEYWORDS:
            if keyword in text_lower:
                results.append((keyword.title(), EntityType.TECHNOLOGY))
        return results

    def _proper_noun_heuristic(self, text: str) -> List[Tuple[str, EntityType]]:
        """Catch multi-word proper nouns that NLTK may have missed."""
        results: List[Tuple[str, EntityType]] = []
        for match in _PROPER_NOUN_RE.finditer(text):
            name = match.group(1)
            if len(name) >= 4 and name.lower() not in _TECH_KEYWORDS:
                results.append((name, EntityType.UNKNOWN))
        return results

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(raw: List[Tuple[str, EntityType]]) -> List[Entity]:
        """Merge duplicates by (lowercased name, type) and count mentions."""
        counter: Counter[Tuple[str, EntityType]] = Counter()
        canonical: Dict[Tuple[str, EntityType], str] = {}

        for name, etype in raw:
            key = (name.lower().strip(), etype)
            counter[key] += 1
            # Keep the first-seen casing as canonical
            if key not in canonical:
                canonical[key] = name.strip()

        entities: List[Entity] = []
        for (_, etype), count in counter.items():
            cname = canonical[(_, etype)]
            entities.append(Entity(name=cname, entity_type=etype, mentions=count))
        return entities
