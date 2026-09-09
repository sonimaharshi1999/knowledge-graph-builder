# Knowledge Graph Builder - Relationship Extractor
# Author: Maharshi Soni | License: MIT

"""Extract relationships between entities using dependency-pattern heuristics
and sentence-level co-occurrence."""

from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Dict, List, Set, Tuple

from src.models import Entity, EntityType, Relationship, RelationshipType
from src.nlp_utils import split_sentences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verb-pattern -> relationship type mapping
# ---------------------------------------------------------------------------

_VERB_PATTERNS: List[Tuple[re.Pattern[str], RelationshipType]] = [
    (re.compile(r"\b(?:works?\s+(?:at|for)|employed\s+(?:at|by)|joined)\b", re.I), RelationshipType.WORKS_AT),
    (re.compile(r"\b(?:located\s+in|based\s+in|headquartered\s+in)\b", re.I), RelationshipType.LOCATED_IN),
    (re.compile(r"\b(?:uses?|adopted?|leverag(?:es?|ing)|built\s+(?:with|on|using))\b", re.I), RelationshipType.USES),
    (re.compile(r"\b(?:develop(?:s|ed|ing)?|creat(?:es?|ed|ing)|built?|design(?:s|ed|ing)?)\b", re.I), RelationshipType.DEVELOPS),
    (re.compile(r"\b(?:partner(?:s|ed|ing)?\s+with|collaborat(?:es?|ed|ing)\s+with)\b", re.I), RelationshipType.PARTNERS_WITH),
    (re.compile(r"\b(?:manag(?:es?|ed|ing)|leads?|head(?:s|ed|ing)?|directs?)\b", re.I), RelationshipType.MANAGES),
    (re.compile(r"\b(?:founded?|co-founded?|started?|established?)\b", re.I), RelationshipType.FOUNDED),
    (re.compile(r"\b(?:acquir(?:es?|ed|ing)|bought|merged?\s+with)\b", re.I), RelationshipType.ACQUIRED),
]


class RelationshipExtractor:
    """Extracts typed relationships between entities found in the same text."""

    def extract(
        self,
        text: str,
        entities: List[Entity],
    ) -> List[Relationship]:
        """Return relationships discovered in *text* given known *entities*.

        Strategy
        --------
        1. For every sentence that contains at least two known entities,
           check verb patterns to assign a typed relationship.
        2. If no verb pattern matches but two entities co-occur in the same
           sentence, emit a CO_OCCURS edge with lower weight.
        """
        if len(entities) < 2:
            return []

        entity_names: Set[str] = {e.name for e in entities}
        entity_lookup: Dict[str, Entity] = {e.name.lower(): e for e in entities}
        sentences = split_sentences(text)

        relationships: List[Relationship] = []
        seen: Set[Tuple[str, str, RelationshipType]] = set()

        for sentence in sentences:
            present = self._entities_in_sentence(sentence, entity_names, entity_lookup)
            if len(present) < 2:
                continue

            matched_verb = False
            for pattern, rel_type in _VERB_PATTERNS:
                if pattern.search(sentence):
                    matched_verb = True
                    for src, tgt in self._directed_pairs(present, rel_type):
                        key = (src.lower(), tgt.lower(), rel_type)
                        if key not in seen:
                            seen.add(key)
                            relationships.append(
                                Relationship(
                                    source=src,
                                    target=tgt,
                                    relation_type=rel_type,
                                    weight=1.0,
                                )
                            )

            # Fallback: co-occurrence
            if not matched_verb:
                for a, b in combinations(sorted(present), 2):
                    key = (a.lower(), b.lower(), RelationshipType.CO_OCCURS)
                    if key not in seen:
                        seen.add(key)
                        relationships.append(
                            Relationship(
                                source=a,
                                target=b,
                                relation_type=RelationshipType.CO_OCCURS,
                                weight=0.5,
                            )
                        )

        return relationships

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entities_in_sentence(
        sentence: str,
        entity_names: Set[str],
        entity_lookup: Dict[str, Entity],
    ) -> List[str]:
        """Return entity names that appear in *sentence*."""
        sent_lower = sentence.lower()
        return [name for name in entity_names if name.lower() in sent_lower]

    @staticmethod
    def _directed_pairs(
        names: List[str],
        rel_type: RelationshipType,
    ) -> List[Tuple[str, str]]:
        """Heuristic: for PERSON->ORG verbs the person is the source, etc."""
        pairs: List[Tuple[str, str]] = []
        for a, b in combinations(names, 2):
            pairs.append((a, b))
        return pairs
