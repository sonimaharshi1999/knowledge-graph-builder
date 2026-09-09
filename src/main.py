# Knowledge Graph Builder - CLI Entry Point
# Author: Maharshi Soni | License: MIT

"""Command-line demo that ingests the synthetic corpus, prints graph
statistics, and runs sample queries."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import List

from src.graph_builder import GraphBuilder
from src.models import DocumentInput
from src.synthetic_data import get_synthetic_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_demo() -> None:
    """Run the full demonstration pipeline."""
    print("=" * 70)
    print("  Knowledge Graph Builder - Demo")
    print("  Author: Maharshi Soni")
    print("=" * 70)
    print()

    # 1. Build the graph
    builder = GraphBuilder()
    documents: List[DocumentInput] = get_synthetic_documents()

    print(f"[1/4] Ingesting {len(documents)} synthetic documents...")
    start = time.perf_counter()
    result = builder.ingest_documents(documents)
    elapsed = time.perf_counter() - start

    print(f"       Done in {elapsed:.3f}s")
    print(f"       Entities extracted:      {result.entities_extracted}")
    print(f"       Relationships extracted: {result.relationships_extracted}")
    print(f"       Graph entities (merged): {result.total_graph_entities}")
    print(f"       Graph edges:             {result.total_graph_relationships}")
    print()

    # 2. Graph statistics
    print("[2/4] Graph statistics:")
    stats = builder.graph.stats()
    print(f"       Nodes:                {stats.total_entities}")
    print(f"       Edges:                {stats.total_relationships}")
    print(f"       Connected components: {stats.connected_components}")
    print(f"       Density:              {stats.density}")
    print(f"       Avg degree:           {stats.avg_degree}")
    print()

    # 3. Sample queries
    print("[3/4] Sample queries:")
    print()

    # Shortest path
    path_result = builder.graph.shortest_path("Alice Chen", "Frank Lee")
    if path_result.exists:
        print(f"  Shortest path Alice Chen -> Frank Lee:")
        print(f"    {' -> '.join(path_result.path)}  (length={path_result.length})")
    else:
        print("  No path found between Alice Chen and Frank Lee")
    print()

    # Neighbors
    neighbors = builder.graph.neighbors("NovaTech", depth=1)
    print(f"  Neighbors of NovaTech (depth=1): {len(neighbors.neighbors)} entities")
    for n in neighbors.neighbors[:5]:
        print(f"    - {n['name']} ({n['entity_type']})")
    if len(neighbors.neighbors) > 5:
        print(f"    ... and {len(neighbors.neighbors) - 5} more")
    print()

    # PageRank
    print("  Top 5 entities by PageRank:")
    for item in builder.graph.pagerank(5):
        print(f"    - {item['entity']}: {item['score']:.6f}")
    print()

    # Connected components
    components = builder.graph.connected_components()
    print(f"  Connected components: {len(components)}")
    for i, comp in enumerate(components[:3]):
        print(f"    Component {i+1} ({len(comp)} nodes): {', '.join(comp[:5])}{'...' if len(comp) > 5 else ''}")
    print()

    # 4. API instructions
    print("[4/4] To start the API server:")
    print("       uvicorn src.api:app --reload")
    print("       Then visit http://localhost:8000/docs for Swagger UI")
    print()
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
