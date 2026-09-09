# Knowledge Graph Builder - Knowledge Graph Engine
# Author: Maharshi Soni | License: MIT

"""NetworkX-backed knowledge graph supporting typed nodes/edges,
graph queries (shortest path, neighbors, PageRank, connected components),
and summary statistics."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.models import (
    Entity,
    EntityType,
    GraphStatsResponse,
    NeighborResponse,
    PathResponse,
    Relationship,
    RelationshipType,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory knowledge graph backed by a NetworkX MultiDiGraph."""

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entities: Dict[str, Entity] = {}  # lowered name -> Entity

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity node."""
        key = entity.name.lower()
        if key in self._entities:
            existing = self._entities[key]
            merged = Entity(
                name=existing.name,
                entity_type=entity.entity_type if entity.entity_type != EntityType.UNKNOWN else existing.entity_type,
                mentions=existing.mentions + entity.mentions,
                metadata={**existing.metadata, **entity.metadata},
            )
            self._entities[key] = merged
            self._graph.nodes[key]["entity_type"] = merged.entity_type.value
            self._graph.nodes[key]["mentions"] = merged.mentions
        else:
            self._entities[key] = entity
            self._graph.add_node(
                key,
                label=entity.name,
                entity_type=entity.entity_type.value,
                mentions=entity.mentions,
            )

    def add_relationship(self, rel: Relationship) -> None:
        """Add a directed, typed edge between two entities."""
        src = rel.source.lower()
        tgt = rel.target.lower()
        # Ensure both endpoints exist
        if src not in self._entities:
            self.add_entity(Entity(name=rel.source, entity_type=EntityType.UNKNOWN))
        if tgt not in self._entities:
            self.add_entity(Entity(name=rel.target, entity_type=EntityType.UNKNOWN))

        self._graph.add_edge(
            src,
            tgt,
            relation_type=rel.relation_type.value,
            weight=rel.weight,
            **rel.metadata,
        )

    def add_entities(self, entities: List[Entity]) -> None:
        """Batch-add entities."""
        for e in entities:
            self.add_entity(e)

    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Batch-add relationships."""
        for r in relationships:
            self.add_relationship(r)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def shortest_path(self, source: str, target: str) -> PathResponse:
        """Find the shortest undirected path between two entities."""
        src = source.lower()
        tgt = target.lower()
        undirected = self._graph.to_undirected()
        try:
            path_nodes = nx.shortest_path(undirected, src, tgt)
            labels = [self._label(n) for n in path_nodes]
            return PathResponse(
                source=self._label(src),
                target=self._label(tgt),
                path=labels,
                length=len(path_nodes) - 1,
                exists=True,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return PathResponse(
                source=source,
                target=target,
                path=[],
                length=0,
                exists=False,
            )

    def neighbors(self, entity: str, depth: int = 1) -> NeighborResponse:
        """Return the neighborhood of *entity* up to *depth* hops."""
        key = entity.lower()
        if key not in self._graph:
            return NeighborResponse(entity=entity, depth=depth, neighbors=[], edges=[])

        undirected = self._graph.to_undirected()
        visited: Set[str] = set()
        frontier: Set[str] = {key}
        all_neighbors: List[Dict[str, Any]] = []
        all_edges: List[Dict[str, Any]] = []

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                for nbr in undirected.neighbors(node):
                    if nbr not in visited and nbr != key:
                        visited.add(nbr)
                        next_frontier.add(nbr)
                        all_neighbors.append(self._node_info(nbr))
                        # Collect edges
                        for _u, _v, data in self._graph.edges(node, data=True):
                            if _v == nbr:
                                all_edges.append({
                                    "source": self._label(node),
                                    "target": self._label(nbr),
                                    "relation_type": data.get("relation_type", "UNKNOWN"),
                                    "weight": data.get("weight", 1.0),
                                })
                        for _u, _v, data in self._graph.edges(nbr, data=True):
                            if _v == node:
                                all_edges.append({
                                    "source": self._label(nbr),
                                    "target": self._label(node),
                                    "relation_type": data.get("relation_type", "UNKNOWN"),
                                    "weight": data.get("weight", 1.0),
                                })
            frontier = next_frontier

        return NeighborResponse(
            entity=self._label(key),
            depth=depth,
            neighbors=all_neighbors,
            edges=all_edges,
        )

    def connected_components(self) -> List[List[str]]:
        """Return connected components as lists of entity labels."""
        undirected = self._graph.to_undirected()
        components = list(nx.connected_components(undirected))
        return [
            sorted(self._label(n) for n in comp)
            for comp in sorted(components, key=len, reverse=True)
        ]

    def pagerank(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the top-N entities by PageRank score."""
        if self._graph.number_of_nodes() == 0:
            return []
        pr = nx.pagerank(self._graph, weight="weight")
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {"entity": self._label(node), "score": round(score, 6)}
            for node, score in ranked
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> GraphStatsResponse:
        """Compute summary statistics for the graph."""
        g = self._graph

        # Entity type counts
        etype_counts: Dict[str, int] = {}
        for _, data in g.nodes(data=True):
            t = data.get("entity_type", "UNKNOWN")
            etype_counts[t] = etype_counts.get(t, 0) + 1

        # Relationship type counts
        rtype_counts: Dict[str, int] = {}
        for _, _, data in g.edges(data=True):
            t = data.get("relation_type", "UNKNOWN")
            rtype_counts[t] = rtype_counts.get(t, 0) + 1

        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        undirected = g.to_undirected()
        n_components = nx.number_connected_components(undirected) if n_nodes > 0 else 0
        density = nx.density(g) if n_nodes > 0 else 0.0
        degrees = [d for _, d in g.degree()]
        avg_deg = float(np.mean(degrees)) if degrees else 0.0

        # Top by degree
        degree_sorted = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:10]
        top_degree = [
            {"entity": self._label(n), "degree": d}
            for n, d in degree_sorted
        ]

        return GraphStatsResponse(
            total_entities=n_nodes,
            total_relationships=n_edges,
            entity_type_counts=etype_counts,
            relationship_type_counts=rtype_counts,
            connected_components=n_components,
            density=round(density, 6),
            avg_degree=round(avg_deg, 2),
            top_entities_by_degree=top_degree,
            top_entities_by_pagerank=self.pagerank(10),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def entity_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def relationship_count(self) -> int:
        return self._graph.number_of_edges()

    def get_entity(self, name: str) -> Optional[Entity]:
        return self._entities.get(name.lower())

    def get_all_entities(self) -> List[Entity]:
        return list(self._entities.values())

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for u, v, data in self._graph.edges(data=True):
            results.append({
                "source": self._label(u),
                "target": self._label(v),
                **data,
            })
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _label(self, node_key: str) -> str:
        """Return the display label for a node key."""
        data = self._graph.nodes.get(node_key, {})
        return data.get("label", node_key)

    def _node_info(self, node_key: str) -> Dict[str, Any]:
        data = self._graph.nodes.get(node_key, {})
        return {
            "name": data.get("label", node_key),
            "entity_type": data.get("entity_type", "UNKNOWN"),
            "mentions": data.get("mentions", 0),
        }
