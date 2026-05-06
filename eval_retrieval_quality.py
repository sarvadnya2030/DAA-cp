#!/usr/bin/env python3
"""
Retrieval Quality Evaluation for Graph Traversal Algorithms

Metrics:
  - Precision@k: |Retrieved ∩ Gold| / |Retrieved|
  - Recall@k:    |Retrieved ∩ Gold| / |Gold|
  - F1@k:        2×(P×R) / (P+R)

Algorithms evaluated:
  1. BFS
  2. DFS
  3. Dijkstra
  4. PPR
  5. SemanticBeam
  6. PST (NEW)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from statistics import mean, median

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics for a single query."""
    query_id: int
    algorithm: str
    top_k: int
    precision: float
    recall: float
    f1: float
    retrieved_count: int
    relevant_count: int
    intersect_count: int


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries."""
    algorithm: str
    top_k: int
    queries: int
    mean_precision: float
    median_precision: float
    mean_recall: float
    median_recall: float
    mean_f1: float
    median_f1: float
    min_f1: float
    max_f1: float


def precision_at_k(retrieved: Set[str], gold: Set[str]) -> float:
    """Precision@k = |Retrieved ∩ Gold| / |Retrieved|"""
    if len(retrieved) == 0:
        return 0.0
    return len(retrieved & gold) / len(retrieved)


def recall_at_k(retrieved: Set[str], gold: Set[str]) -> float:
    """Recall@k = |Retrieved ∩ Gold| / |Gold|"""
    if len(gold) == 0:
        return 0.0
    return len(retrieved & gold) / len(gold)


def f1_at_k(precision: float, recall: float) -> float:
    """F1@k = 2×(P×R) / (P+R)"""
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def evaluate_query(
    retrieved: Set[str],
    gold: Set[str],
    query_id: int,
    algorithm: str,
    top_k: int,
) -> RetrievalMetrics:
    """Evaluate a single query's retrieval results."""
    p = precision_at_k(retrieved, gold)
    r = recall_at_k(retrieved, gold)
    f1 = f1_at_k(p, r)

    return RetrievalMetrics(
        query_id=query_id,
        algorithm=algorithm,
        top_k=top_k,
        precision=p,
        recall=r,
        f1=f1,
        retrieved_count=len(retrieved),
        relevant_count=len(gold),
        intersect_count=len(retrieved & gold),
    )


# ─────────────────────────────────────────────────────────────────────
# Test Data Generation
# ─────────────────────────────────────────────────────────────────────

def generate_synthetic_test_set(
    graph: nx.Graph,
    n_queries: int = 20,
    n_gold_per_query: int = 5,
) -> List[Dict]:
    """
    Generate synthetic test set: for each query, pick:
      - Random seed nodes (1-3)
      - Gold relevant nodes (actual neighbors of seeds)

    Assumption: nodes within 2 hops of seed are "relevant"
    """
    queries = []
    nodes = list(graph.nodes())
    rng = np.random.RandomState(42)

    for query_id in range(n_queries):
        # Pick random seeds
        n_seeds = rng.randint(1, 4)
        seed_nodes = set(rng.choice(nodes, size=n_seeds, replace=False).tolist())

        # Collect all nodes within 2 hops of seeds (gold standard)
        gold_nodes: Set[str] = set()
        visited = set(seed_nodes)

        for seed in seed_nodes:
            if seed not in graph:
                continue
            # Hop 1
            for hop1 in graph[seed]:
                gold_nodes.add(hop1)
                visited.add(hop1)
            # Hop 2
            for hop1 in list(graph[seed]):
                if hop1 in graph:
                    for hop2 in graph[hop1]:
                        if hop2 not in seed_nodes:
                            gold_nodes.add(hop2)

        # Limit gold to top n_gold (by relevance = edge weight sum)
        if len(gold_nodes) > n_gold_per_query:
            # Score by proximity to seeds
            scores = {}
            for node in gold_nodes:
                weight_sum = 0.0
                for seed in seed_nodes:
                    if seed in graph and node in graph[seed]:
                        weight_sum += graph[seed][node].get("weight", 0.0)
                    # Check hop-2 distance
                    if seed in graph:
                        for hop1 in graph[seed]:
                            if hop1 in graph and node in graph[hop1]:
                                weight_sum += 0.5 * graph[hop1][node].get("weight", 0.0)
                scores[node] = weight_sum

            sorted_gold = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            gold_nodes = set(node for node, _ in sorted_gold[:n_gold_per_query])

        queries.append({
            "query_id": query_id,
            "seed_nodes": list(seed_nodes),
            "gold_nodes": list(gold_nodes),
        })

    return queries


# ─────────────────────────────────────────────────────────────────────
# Baseline Algorithm Implementations (from benchmark_graph_traversers.py)
# ─────────────────────────────────────────────────────────────────────

def bfs_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> Set[str]:
    """BFS retrieval."""
    visited: Set[str] = set(seeds)
    candidates: Set[str] = set()

    for seed in seeds:
        if seed not in graph:
            continue
        for neighbor in graph[seed]:
            if neighbor not in visited:
                candidates.add(neighbor)
                visited.add(neighbor)

    for node in list(candidates):
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    candidates.add(neighbor)
                    visited.add(neighbor)

    # Score by edge weight sum
    scores = {}
    for node in candidates:
        weight_sum = 0.0
        for seed in seeds:
            if seed in graph and node in graph[seed]:
                weight_sum += graph[seed][node].get("weight", 0.0)
        scores[node] = weight_sum

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return set(node for node, _ in sorted_results[:top_k])


def dfs_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> Set[str]:
    """DFS retrieval."""
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

    scores = {node: float(graph.degree(node)) for node in candidates if node in graph}
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return set(node for node, _ in sorted_results[:top_k])


def dijkstra_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> Set[str]:
    """Dijkstra retrieval."""
    all_distances: Dict[str, float] = {}

    for seed in seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
            for node, dist in lengths.items():
                if node not in all_distances or dist < all_distances[node]:
                    all_distances[node] = dist
        except (nx.NodeNotFound, nx.NetworkXError):
            continue

    candidates = {
        node: 1.0 / (1.0 + dist)
        for node, dist in all_distances.items()
        if node not in seeds
    }

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return set(node for node, _ in sorted_results[:top_k])


def ppr_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> Set[str]:
    """PPR retrieval."""
    if graph.number_of_nodes() == 0:
        return set()

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

    candidates = {
        node: score for node, score in ppr_scores.items()
        if node not in seeds
    }

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return set(node for node, _ in sorted_results[:top_k])


def semantic_beam_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
) -> Set[str]:
    """SemanticBeam retrieval."""
    if query_emb is None or node_embeddings is None:
        return bfs_traverse(graph, seeds, top_k)

    visited: Set[str] = set(seeds)
    frontier: List[str] = list(seeds)
    all_candidates: Dict[str, float] = {}
    beam_width = 10

    for iteration in range(3):
        next_frontier: Dict[str, float] = {}

        for node in frontier:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                if neighbor in node_embeddings:
                    sim = float(np.dot(query_emb, node_embeddings[neighbor].T))
                    sim = max(0.0, sim)
                else:
                    sim = 0.0

                next_frontier[neighbor] = sim
                all_candidates[neighbor] = sim

        sorted_next = sorted(next_frontier.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        frontier = [node for node, _ in sorted_next]

        if not frontier:
            break

    sorted_results = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    return set(node for node, _ in sorted_results[:top_k])


def pst_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    query_emb: Optional[np.ndarray] = None,
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    top_k: int = 10,
) -> Set[str]:
    """PST retrieval."""
    from core.pst_traverser import PSTTraverser

    if node_embeddings is None or not node_embeddings:
        return bfs_traverse(graph, seeds, top_k)

    traverser = PSTTraverser(k_prune=15)
    results = traverser.traverse(
        graph,
        seeds,
        query="",
        node_embeddings=node_embeddings,
        top_k=top_k,
    )

    return set(node for node, _ in results)


# ─────────────────────────────────────────────────────────────────────
# Evaluation Driver
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--nodes", type=int, default=250, help="Graph size")
    parser.add_argument("--density", type=float, default=0.015, help="Graph density")
    parser.add_argument("--queries", type=int, default=20, help="Number of test queries")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth")
    parser.add_argument("--gold-per-query", type=int, default=5, help="Gold nodes per query")
    parser.add_argument("--output", default="eval_retrieval_results.json", help="Output file")
    args = parser.parse_args()

    print("=" * 80)
    print("Retrieval Quality Evaluation — Precision@k, Recall@k, F1@k")
    print("=" * 80)

    # Build graph
    print(f"\nBuilding test graph ({args.nodes} nodes, density={args.density})...")
    g = nx.Graph()
    nodes = [f"node_{i}" for i in range(args.nodes)]
    g.add_nodes_from(nodes)

    rng = np.random.RandomState(42)
    edge_count = int(args.nodes * (args.nodes - 1) / 2 * args.density)
    for _ in range(edge_count):
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and not g.has_edge(u, v):
            weight = float(rng.exponential(0.5))
            g.add_edge(u, v, weight=weight)

    print(f"✓ Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Generate embeddings
    embeddings = {}
    for node in g.nodes():
        emb = rng.randn(1, 768).astype(np.float32)
        norm = np.linalg.norm(emb)
        embeddings[node] = emb / norm if norm > 0 else emb

    # Generate test queries
    print(f"\nGenerating {args.queries} test queries...")
    test_queries = generate_synthetic_test_set(g, args.queries, args.gold_per_query)
    print(f"✓ Generated {len(test_queries)} queries")

    # Define algorithms
    algorithms = [
        ("BFS", lambda seeds, query_emb, embs: bfs_traverse(g, seeds, args.top_k)),
        ("DFS", lambda seeds, query_emb, embs: dfs_traverse(g, seeds, args.top_k)),
        ("Dijkstra", lambda seeds, query_emb, embs: dijkstra_traverse(g, seeds, args.top_k)),
        ("PPR", lambda seeds, query_emb, embs: ppr_traverse(g, seeds, args.top_k)),
        ("SemanticBeam", lambda seeds, query_emb, embs: semantic_beam_traverse(g, seeds, query_emb, embs, args.top_k)),
        ("PST", lambda seeds, query_emb, embs: pst_traverse(g, seeds, query_emb, embs, args.top_k)),
    ]

    results: Dict[str, List[RetrievalMetrics]] = {algo_name: [] for algo_name, _ in algorithms}

    # Evaluate each algorithm
    for algo_name, algo_fn in algorithms:
        print(f"\nEvaluating {algo_name}...")

        for query_data in test_queries:
            seeds = set(query_data["seed_nodes"])
            gold = set(query_data["gold_nodes"])
            query_id = query_data["query_id"]

            # Generate random query embedding
            query_emb = rng.randn(1, 768).astype(np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Run algorithm
            retrieved = algo_fn(seeds, query_emb, embeddings)

            # Evaluate
            metrics = evaluate_query(retrieved, gold, query_id, algo_name, args.top_k)
            results[algo_name].append(metrics)

        # Print first query result
        first = results[algo_name][0]
        print(
            f"  Query 1: P@{args.top_k}={first.precision:.3f}, "
            f"R@{args.top_k}={first.recall:.3f}, F1={first.f1:.3f}"
        )

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    aggregate_results: Dict[str, AggregateMetrics] = {}

    print(f"\n{'Algorithm':<15} {'P@{0}':^12} {'R@{0}':^12} {'F1@{0}':^12}".format(args.top_k))
    print("-" * 80)

    for algo_name, metrics_list in results.items():
        precisions = [m.precision for m in metrics_list]
        recalls = [m.recall for m in metrics_list]
        f1s = [m.f1 for m in metrics_list]

        agg = AggregateMetrics(
            algorithm=algo_name,
            top_k=args.top_k,
            queries=len(metrics_list),
            mean_precision=float(mean(precisions)),
            median_precision=float(median(precisions)),
            mean_recall=float(mean(recalls)),
            median_recall=float(median(recalls)),
            mean_f1=float(mean(f1s)),
            median_f1=float(median(f1s)),
            min_f1=float(min(f1s)),
            max_f1=float(max(f1s)),
        )
        aggregate_results[algo_name] = agg

        marker = "⭐" if algo_name == "PST" else "  "
        print(
            f"{marker} {algo_name:<13} "
            f"{agg.mean_precision:<12.3f} "
            f"{agg.mean_recall:<12.3f} "
            f"{agg.mean_f1:<12.3f}"
        )

    # Save results
    output_data = {
        "meta": {
            "graph_nodes": g.number_of_nodes(),
            "graph_edges": g.number_of_edges(),
            "top_k": args.top_k,
            "queries": args.queries,
            "gold_per_query": args.gold_per_query,
        },
        "per_query": {
            algo_name: [asdict(m) for m in metrics_list]
            for algo_name, metrics_list in results.items()
        },
        "aggregate": {
            algo_name: asdict(agg)
            for algo_name, agg in aggregate_results.items()
        },
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n✓ Results saved to {output_path}")

    # Print F1 ranking
    print("\n" + "=" * 80)
    print("F1@{0} RANKING".format(args.top_k))
    print("=" * 80)
    ranked = sorted(
        aggregate_results.items(),
        key=lambda x: x[1].mean_f1,
        reverse=True,
    )
    for rank, (algo_name, agg) in enumerate(ranked, 1):
        marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"  {rank}."
        print(
            f"{marker} {algo_name:<15} F1={agg.mean_f1:.4f} "
            f"(P={agg.mean_precision:.3f}, R={agg.mean_recall:.3f})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
