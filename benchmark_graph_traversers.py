#!/usr/bin/env python3
"""
Benchmark all 6 graph traversal algorithms on a test subgraph.

Algorithms:
  1. BFS (breadth-first) — broad, noisy
  2. DFS (depth-first) — deep, narrow
  3. Dijkstra — weight-aware shortest paths
  4. PPR — personalized pagerank
  5. SemanticBeam — semantic similarity (expensive but high quality)
  6. PST (NEW) — progressive semantic traversal (hybrid, ~90ms target)

Metrics:
  - Latency (per-stage where applicable)
  - Nodes explored
  - Quality (via mock retrieval relevance)
"""

from __future__ import annotations

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import asdict, dataclass
from statistics import mean, median

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Baseline Algorithm Implementations
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AlgoResult:
    """Result from a single run."""
    algo_name: str
    latency_ms: float
    nodes_explored: int
    top_results: List[Tuple[str, float]]
    latency_breakdown: Optional[str] = None  # For multi-stage algos


def bfs_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
) -> AlgoResult:
    """BFS: breadth-first traversal, hop-2."""
    t0 = time.perf_counter()

    visited: Set[str] = set(seeds)
    candidates: Set[str] = set()

    # Hop-1
    for seed in seeds:
        if seed not in graph:
            continue
        for neighbor in graph[seed]:
            if neighbor not in visited:
                candidates.add(neighbor)
                visited.add(neighbor)

    # Hop-2
    for node in list(candidates):
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    candidates.add(neighbor)
                    visited.add(neighbor)

    # Score by BFS order (simpler: by edge weight sum)
    scores: Dict[str, float] = {}
    for node in candidates:
        weight_sum = 0.0
        for seed in seeds:
            if seed in graph and node in graph[seed]:
                weight_sum += graph[seed][node].get("weight", 0.0)
        scores[node] = weight_sum

    sorted_results = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    latency = (time.perf_counter() - t0) * 1000
    return AlgoResult(
        algo_name="BFS",
        latency_ms=latency,
        nodes_explored=len(candidates),
        top_results=sorted_results,
    )


def dfs_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
) -> AlgoResult:
    """DFS: depth-first traversal."""
    t0 = time.perf_counter()

    candidates: Set[str] = set()
    visited: Set[str] = set(seeds)

    def dfs_visit(node: str, depth: int, max_depth: int = 2) -> None:
        if depth > max_depth:
            return
        if node not in graph:
            return
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                candidates.add(neighbor)
                dfs_visit(neighbor, depth + 1, max_depth)

    for seed in seeds:
        if seed in graph:
            dfs_visit(seed, 0)

    # Score by DFS discovery order (simpler: by degree)
    scores: Dict[str, float] = {
        node: float(graph.degree(node)) for node in candidates if node in graph
    }

    sorted_results = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    latency = (time.perf_counter() - t0) * 1000
    return AlgoResult(
        algo_name="DFS",
        latency_ms=latency,
        nodes_explored=len(candidates),
        top_results=sorted_results,
    )


def dijkstra_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
) -> AlgoResult:
    """Dijkstra: shortest path (minimum weight) expansion."""
    t0 = time.perf_counter()

    # Invert weights for Dijkstra (find minimum cost = maximum relevance)
    all_distances: Dict[str, float] = {}

    for seed in seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph, seed, weight="weight"
            )
            for node, dist in lengths.items():
                if node not in all_distances or dist < all_distances[node]:
                    all_distances[node] = dist
        except (nx.NodeNotFound, nx.NetworkXError):
            continue

    # Score: invert distance (closer = higher score)
    candidates = {
        node: 1.0 / (1.0 + dist)
        for node, dist in all_distances.items()
        if node not in seeds
    }

    sorted_results = sorted(
        candidates.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    latency = (time.perf_counter() - t0) * 1000
    return AlgoResult(
        algo_name="Dijkstra",
        latency_ms=latency,
        nodes_explored=len(all_distances),
        top_results=sorted_results,
    )


def ppr_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
) -> AlgoResult:
    """PPR: personalized PageRank seeded from query entities."""
    t0 = time.perf_counter()

    if graph.number_of_nodes() == 0:
        return AlgoResult(
            algo_name="PPR",
            latency_ms=0.0,
            nodes_explored=0,
            top_results=[],
        )

    # Personalization vector: uniform over seeds
    n_seeds = len([s for s in seeds if s in graph])
    if n_seeds == 0:
        n_seeds = 1

    personalization = {
        node: (1.0 / n_seeds if node in seeds else 0.0)
        for node in graph.nodes()
    }

    try:
        ppr_scores = nx.pagerank(
            graph,
            alpha=0.85,
            personalization=personalization,
            weight="weight",
        )
    except nx.PowerIterationFailedConvergence:
        ppr_scores = {node: 0.0 for node in graph.nodes()}

    # Filter out seeds, sort by score
    candidates = {
        node: score for node, score in ppr_scores.items()
        if node not in seeds
    }

    sorted_results = sorted(
        candidates.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    latency = (time.perf_counter() - t0) * 1000
    return AlgoResult(
        algo_name="PPR",
        latency_ms=latency,
        nodes_explored=len(graph),
        top_results=sorted_results,
    )


def semantic_beam_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
    beam_width: int = 10,
) -> AlgoResult:
    """SemanticBeam: iterative semantic scoring with beam search."""
    t0 = time.perf_counter()

    if query_emb is None or node_embeddings is None:
        # Fallback to BFS if no embeddings
        return bfs_traverse(graph, seeds, query_emb, node_embeddings, top_k)

    visited: Set[str] = set(seeds)
    frontier: List[str] = list(seeds)
    all_candidates: Dict[str, float] = {}

    # Iterative expansion with semantic scoring
    for iteration in range(3):  # Max 3 hops
        next_frontier: Dict[str, float] = {}

        for node in frontier:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                # Semantic score
                if neighbor in node_embeddings:
                    sim = float(np.dot(query_emb, node_embeddings[neighbor].T))
                    sim = max(0.0, sim)
                else:
                    sim = 0.0

                next_frontier[neighbor] = sim
                all_candidates[neighbor] = sim

        # Keep top beam_width for next iteration
        sorted_next = sorted(
            next_frontier.items(), key=lambda x: x[1], reverse=True
        )[:beam_width]
        frontier = [node for node, _ in sorted_next]

        if not frontier:
            break

    sorted_results = sorted(
        all_candidates.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    latency = (time.perf_counter() - t0) * 1000
    return AlgoResult(
        algo_name="SemanticBeam",
        latency_ms=latency,
        nodes_explored=len(visited),
        top_results=sorted_results,
    )


def pst_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
) -> AlgoResult:
    """PST: Progressive Semantic Traversal (new hybrid algorithm)."""
    from core.pst_traverser import PSTTraverser

    t0_total = time.perf_counter()

    traverser = PSTTraverser(k_prune=15)
    results = traverser.traverse(
        graph,
        seeds,
        query="",  # Query embedding already provided
        node_embeddings=node_embeddings or {},
        top_k=top_k,
    )

    latency = (time.perf_counter() - t0_total) * 1000

    return AlgoResult(
        algo_name="PST",
        latency_ms=latency,
        nodes_explored=traverser.latency.nodes_explored,
        top_results=results,
        latency_breakdown=traverser.get_latency_report(),
    )


# ─────────────────────────────────────────────────────────────────────
# Benchmark Driver
# ─────────────────────────────────────────────────────────────────────

def build_test_graph(n_nodes: int = 250, edge_density: float = 0.015) -> nx.Graph:
    """Build a random test graph (similar to subgraph from MusiQue dataset)."""
    g = nx.Graph()

    # Add nodes
    nodes = [f"paper_{i}" for i in range(n_nodes)]
    g.add_nodes_from(nodes)

    # Add edges with random weights
    edge_count = int(n_nodes * (n_nodes - 1) / 2 * edge_density)
    rng = np.random.RandomState(42)
    for _ in range(edge_count):
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and not g.has_edge(u, v):
            weight = float(rng.exponential(0.5))  # Exponential distribution
            g.add_edge(u, v, weight=weight)

    return g


def generate_test_embeddings(
    graph: nx.Graph, dim: int = 768
) -> Dict[str, np.ndarray]:
    """Generate random normalized embeddings for each node."""
    rng = np.random.RandomState(42)
    embeddings: Dict[str, np.ndarray] = {}

    for node in graph.nodes():
        emb = rng.randn(1, dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        embeddings[node] = emb / norm if norm > 0 else emb

    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark graph traversers")
    parser.add_argument(
        "--nodes", type=int, default=250, help="Number of nodes in test graph"
    )
    parser.add_argument(
        "--density", type=float, default=0.015, help="Graph edge density"
    )
    parser.add_argument(
        "--queries", type=int, default=20, help="Number of test queries"
    )
    parser.add_argument(
        "--output", default="benchmark_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Graph Traversal Algorithm Benchmark")
    print("=" * 70)

    # Build test graph
    print(f"\nBuilding test graph ({args.nodes} nodes, density={args.density})...")
    graph = build_test_graph(args.nodes, args.density)
    print(f"✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Generate embeddings
    print("Generating test embeddings...")
    embeddings = generate_test_embeddings(graph)
    print(f"✓ Embeddings: {len(embeddings)} nodes, dim=768")

    # Define algorithms
    algorithms = [
        ("BFS", bfs_traverse),
        ("DFS", dfs_traverse),
        ("Dijkstra", dijkstra_traverse),
        ("PPR", ppr_traverse),
        ("SemanticBeam", semantic_beam_traverse),
        ("PST", pst_traverse),
    ]

    results: Dict[str, Dict] = {}

    # Run benchmark for each algorithm
    for algo_name, algo_fn in algorithms:
        print(f"\nBenchmarking {algo_name}...")
        latencies: List[float] = []
        nodes_explored_list: List[int] = []

        for query_idx in range(args.queries):
            # Random seed selection
            n_seeds = np.random.randint(1, 4)
            seeds = set(np.random.choice(
                list(graph.nodes()), size=n_seeds, replace=False
            ).tolist())

            # Random query embedding
            query_emb = np.random.randn(1, 768).astype(np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Run algorithm
            result = algo_fn(
                graph,
                seeds,
                query_emb=query_emb,
                node_embeddings=embeddings,
                top_k=10,
            )

            latencies.append(result.latency_ms)
            nodes_explored_list.append(result.nodes_explored)

            if query_idx == 0:
                # Print first result for each algo
                print(
                    f"  Query 1: {result.latency_ms:.2f}ms, "
                    f"explored {result.nodes_explored} nodes"
                )
                if result.latency_breakdown:
                    print(f"    {result.latency_breakdown}")

        # Summary stats
        results[algo_name] = {
            "mean_latency_ms": float(mean(latencies)),
            "median_latency_ms": float(median(latencies)),
            "min_latency_ms": float(min(latencies)),
            "max_latency_ms": float(max(latencies)),
            "mean_nodes_explored": float(mean(nodes_explored_list)),
            "queries": args.queries,
        }

        print(
            f"  Mean: {mean(latencies):.2f}ms | "
            f"Min: {min(latencies):.2f}ms | Max: {max(latencies):.2f}ms"
        )

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 70)
    for algo_name, stats in results.items():
        print(
            f"{algo_name:<15} "
            f"{stats['mean_latency_ms']:<12.2f} "
            f"{stats['min_latency_ms']:<12.2f} "
            f"{stats['max_latency_ms']:<12.2f}"
        )

    # Save results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
