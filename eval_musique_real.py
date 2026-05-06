#!/usr/bin/env python3
"""
Comprehensive evaluation of 6 graph traversal algorithms on real MuSiQue multi-hop dataset.

Algorithms:
  1. BFS (Breadth-First Search)
  2. DFS (Depth-First Search)
  3. Dijkstra (Shortest Path)
  4. PPR (Personalized PageRank)
  5. SemanticBeam (Semantic Similarity)
  6. PST (Progressive Semantic Traversal)

Metrics: P@k, R@k, F1@k, MRR, NDCG@k, Hit@k

Dataset: Real MuSiQue graph (6,287 nodes, 74,057 edges), 500 multi-hop questions
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev
import sys

import numpy as np
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Per-question metrics."""
    question_id: str
    algorithm: str
    latency_ms: float
    precision_at_10: float
    recall_at_10: float
    f1_at_10: float
    mrr: float
    ndcg_at_10: float
    hit_at_10: int
    retrieved_count: int
    gold_count: int
    intersect_count: int


@dataclass
class AggregateMetrics:
    """Aggregated metrics across questions."""
    algorithm: str
    questions: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_mrr: float
    mean_ndcg: float
    mean_hit_rate: float
    mean_latency_ms: float
    median_latency_ms: float


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: Set[str], gold: Set[str], k: int = 10) -> float:
    """Precision@k: |retrieved ∩ gold| / |retrieved|"""
    if len(retrieved) == 0:
        return 0.0
    return len(retrieved & gold) / min(len(retrieved), k)


def recall_at_k(retrieved: Set[str], gold: Set[str], k: int = 10) -> float:
    """Recall@k: |retrieved ∩ gold| / |gold|"""
    if len(gold) == 0:
        return 0.0
    return len(retrieved & gold) / len(gold)


def f1_at_k(precision: float, recall: float) -> float:
    """F1@k: Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def mrr(retrieved_ordered: List[str], gold: Set[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first gold match."""
    for rank, entity in enumerate(retrieved_ordered[:10], start=1):
        if entity in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ordered: List[str], gold: Set[str], k: int = 10) -> float:
    """NDCG@k: Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    idcg = 0.0

    for rank, entity in enumerate(retrieved_ordered[:k], start=1):
        if entity in gold:
            dcg += 1.0 / np.log2(rank + 1)

    for rank in range(1, min(len(gold) + 1, k + 1)):
        idcg += 1.0 / np.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(retrieved: Set[str], gold: Set[str], k: int = 10) -> int:
    """Hit@k: 1 if any gold in top-k, 0 otherwise."""
    return 1 if len(retrieved & gold) > 0 else 0


# ─────────────────────────────────────────────────────────────────────
# Baseline Algorithms
# ─────────────────────────────────────────────────────────────────────

def bfs_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> List[str]:
    """BFS: Breadth-first search up to 2 hops."""
    visited: Set[str] = set(seeds)
    candidates: Dict[str, float] = {}

    # Hop 1
    for seed in seeds:
        if seed not in graph:
            continue
        for neighbor in graph[seed]:
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor not in candidates:
                    candidates[neighbor] = 0.0
                candidates[neighbor] += float(graph[seed][neighbor].get("weight", 1.0))

    # Hop 2
    for node in list(candidates.keys()):
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor not in candidates:
                        candidates[neighbor] = 0.0
                    candidates[neighbor] += 0.5 * float(graph[node][neighbor].get("weight", 1.0))

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def dfs_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> List[str]:
    """DFS: Depth-first search up to 2 hops."""
    visited: Set[str] = set(seeds)
    candidates: Dict[str, float] = {}

    def dfs_visit(node: str, depth: int = 0, max_depth: int = 2, weight_decay: float = 1.0):
        if depth > max_depth or node not in graph:
            return
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor not in candidates:
                    candidates[neighbor] = 0.0
                candidates[neighbor] += weight_decay * float(graph[node][neighbor].get("weight", 1.0))
                dfs_visit(neighbor, depth + 1, max_depth, weight_decay * 0.7)

    for seed in seeds:
        dfs_visit(seed, 0)

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def dijkstra_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> List[str]:
    """Dijkstra: Shortest path expansion (minimum weight = maximum similarity)."""
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

    # Score: invert distance (closer = higher score), filter seeds
    candidates = {
        node: 1.0 / (1.0 + dist)
        for node, dist in all_distances.items()
        if node not in seeds and dist > 0
    }

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def ppr_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    top_k: int = 10,
) -> List[str]:
    """PPR: Personalized PageRank seeded from query entities."""
    if graph.number_of_nodes() == 0:
        return []

    # Personalization vector: uniform over seeds
    n_seeds = len([s for s in seeds if s in graph])
    if n_seeds == 0:
        return []

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
            max_iter=100,
        )
    except nx.PowerIterationFailedConvergence:
        ppr_scores = {node: 0.0 for node in graph.nodes()}

    # Filter out seeds, sort by score
    candidates = {
        node: score for node, score in ppr_scores.items()
        if node not in seeds
    }

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def semantic_beam_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    query_emb: Optional[np.ndarray] = None,
    top_k: int = 10,
    beam_width: int = 15,
) -> List[str]:
    """SemanticBeam: Iterative semantic scoring with beam search."""
    if node_embeddings is None or query_emb is None:
        return bfs_traverse(graph, seeds, top_k)

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

                # Semantic similarity score
                if neighbor in node_embeddings:
                    sim = float(np.dot(query_emb, node_embeddings[neighbor].T).flatten()[0])
                    sim = max(0.0, sim)
                else:
                    sim = 0.0

                next_frontier[neighbor] = sim
                all_candidates[neighbor] = max(all_candidates.get(neighbor, 0.0), sim)

        # Keep top beam_width for next iteration
        sorted_next = sorted(next_frontier.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        frontier = [node for node, _ in sorted_next]

        if not frontier:
            break

    sorted_results = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def pst_traverse(
    graph: nx.Graph,
    seeds: Set[str],
    node_embeddings: Optional[Dict[str, np.ndarray]] = None,
    query_emb: Optional[np.ndarray] = None,
    top_k: int = 10,
) -> List[str]:
    """PST: Progressive Semantic Traversal (hybrid)."""
    try:
        from core.pst_traverser import PSTTraverser
    except ImportError:
        logger.warning("PST not available, using BFS fallback")
        return bfs_traverse(graph, seeds, top_k)

    if node_embeddings is None:
        return bfs_traverse(graph, seeds, top_k)

    traverser = PSTTraverser(k_prune=40)
    results = traverser.traverse(
        graph,
        seeds,
        query="",
        node_embeddings=node_embeddings,
        top_k=top_k,
    )

    return [node for node, _ in results]


# ─────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate 6 algorithms on real MuSiQue")
    parser.add_argument("--graph-path", default="musique_graph/graph.pkl", help="Path to cached graph")
    parser.add_argument("--embeddings-path", default="musique_graph/embeddings.pkl", help="Path to embeddings")
    parser.add_argument("--questions-path", default="musique_graph/test_questions.pkl", help="Path to test questions")
    parser.add_argument("--max-questions", type=int, default=100, help="Max questions to evaluate")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for retrieval")
    parser.add_argument("--output", default="eval_musique_real_results.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 90)
    print("COMPREHENSIVE EVALUATION: 6 Algorithms on Real MuSiQue Multi-Hop Dataset")
    print("=" * 90)

    # Load graph
    print(f"\n[1/4] Loading graph from {args.graph_path}...")
    with open(args.graph_path, "rb") as f:
        graph = pickle.load(f)
    print(f"✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Load embeddings
    print(f"\n[2/4] Loading embeddings from {args.embeddings_path}...")
    with open(args.embeddings_path, "rb") as f:
        embeddings_data = pickle.load(f)

    # Convert embeddings dict format if needed
    if isinstance(embeddings_data, dict):
        node_embeddings = embeddings_data
    else:
        node_embeddings = {}

    print(f"✓ Embeddings: {len(node_embeddings)} nodes")

    # Load test questions
    print(f"\n[3/4] Loading test questions from {args.questions_path}...")
    with open(args.questions_path, "rb") as f:
        all_questions = pickle.load(f)

    # Limit to max_questions
    questions = all_questions[:args.max_questions]
    print(f"✓ Questions: {len(questions)} multi-hop questions (from {len(all_questions)} total)")

    # Define algorithms
    algorithms = [
        ("BFS", lambda g, s, emb, q: bfs_traverse(g, s, args.top_k)),
        ("DFS", lambda g, s, emb, q: dfs_traverse(g, s, args.top_k)),
        ("Dijkstra", lambda g, s, emb, q: dijkstra_traverse(g, s, args.top_k)),
        ("PPR", lambda g, s, emb, q: ppr_traverse(g, s, args.top_k)),
        ("SemanticBeam", lambda g, s, emb, q: semantic_beam_traverse(g, s, emb, q, args.top_k)),
        ("PST", lambda g, s, emb, q: pst_traverse(g, s, emb, q, args.top_k)),
    ]

    all_results: Dict[str, List[RetrievalMetrics]] = {name: [] for name, _ in algorithms}

    # Evaluate
    print(f"\n[4/4] Evaluating {len(algorithms)} algorithms on {len(questions)} questions...")
    print("-" * 90)

    for algo_idx, (algo_name, algo_fn) in enumerate(algorithms, 1):
        logger.info(f"\n  [{algo_idx}/{len(algorithms)}] {algo_name}")

        latencies = []
        precisions = []
        recalls = []
        f1s = []
        mrrs = []
        ndcgs = []
        hits = []

        for q_idx, q_data in enumerate(questions):
            question_id = q_data["id"]
            gold_titles = set(q_data["gold_titles"])
            all_titles = set(q_data["all_titles"])

            # Use first 1-3 titles as seeds (simulating question entity extraction)
            n_seeds = min(np.random.randint(1, 3), len(all_titles))
            seeds = set(np.random.choice(list(all_titles), size=n_seeds, replace=False).tolist())

            # Random query embedding (simulating query vector)
            query_emb = np.random.randn(1, 768).astype(np.float32)
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

            # Run algorithm
            t0 = time.perf_counter()
            retrieved = algo_fn(graph, seeds, node_embeddings, query_emb)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Convert to set for metrics
            retrieved_set = set(retrieved[:args.top_k])

            # Compute metrics
            p = precision_at_k(retrieved_set, gold_titles, args.top_k)
            r = recall_at_k(retrieved_set, gold_titles, args.top_k)
            f1 = f1_at_k(p, r)
            mrr_score = mrr(retrieved, gold_titles)
            ndcg_score = ndcg_at_k(retrieved, gold_titles, args.top_k)
            hit = hit_at_k(retrieved_set, gold_titles, args.top_k)

            metrics = RetrievalMetrics(
                question_id=question_id,
                algorithm=algo_name,
                latency_ms=latency_ms,
                precision_at_10=p,
                recall_at_10=r,
                f1_at_10=f1,
                mrr=mrr_score,
                ndcg_at_10=ndcg_score,
                hit_at_10=hit,
                retrieved_count=len(retrieved_set),
                gold_count=len(gold_titles),
                intersect_count=len(retrieved_set & gold_titles),
            )
            all_results[algo_name].append(metrics)

            latencies.append(latency_ms)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            mrrs.append(mrr_score)
            ndcgs.append(ndcg_score)
            hits.append(hit)

            if (q_idx + 1) % 20 == 0 or q_idx == 0:
                logger.info(
                    f"    Q{q_idx+1}: P={p:.3f} R={r:.3f} F1={f1:.3f} "
                    f"MRR={mrr_score:.3f} NDCG={ndcg_score:.3f} ({latency_ms:.1f}ms)"
                )

        logger.info(
            f"    Summary: P={mean(precisions):.3f} R={mean(recalls):.3f} F1={mean(f1s):.3f} "
            f"MRR={mean(mrrs):.3f} NDCG={mean(ndcgs):.3f} Hit={sum(hits)}/{len(hits)} "
            f"Latency={mean(latencies):.1f}ms (σ={stdev(latencies) if len(latencies) > 1 else 0:.1f})"
        )

    # Aggregate results
    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)
    print(f"\n{'Algorithm':<18} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'MRR':<8} {'NDCG':<8} {'Hit':<8} {'Latency(ms)':<12}")
    print("-" * 90)

    aggregate: Dict[str, AggregateMetrics] = {}

    for algo_name, metrics_list in all_results.items():
        precisions = [m.precision_at_10 for m in metrics_list]
        recalls = [m.recall_at_10 for m in metrics_list]
        f1s = [m.f1_at_10 for m in metrics_list]
        mrrs = [m.mrr for m in metrics_list]
        ndcgs = [m.ndcg_at_10 for m in metrics_list]
        hits = [m.hit_at_10 for m in metrics_list]
        latencies = [m.latency_ms for m in metrics_list]

        agg = AggregateMetrics(
            algorithm=algo_name,
            questions=len(metrics_list),
            mean_precision=float(mean(precisions)),
            mean_recall=float(mean(recalls)),
            mean_f1=float(mean(f1s)),
            mean_mrr=float(mean(mrrs)),
            mean_ndcg=float(mean(ndcgs)),
            mean_hit_rate=float(sum(hits)) / len(hits),
            mean_latency_ms=float(mean(latencies)),
            median_latency_ms=float(median(latencies)),
        )
        aggregate[algo_name] = agg

        marker = "⭐" if algo_name in ["PST", "SemanticBeam"] else "  "
        print(
            f"{marker}{algo_name:<16} "
            f"{agg.mean_precision:<8.3f} "
            f"{agg.mean_recall:<8.3f} "
            f"{agg.mean_f1:<8.3f} "
            f"{agg.mean_mrr:<8.3f} "
            f"{agg.mean_ndcg:<8.3f} "
            f"{agg.mean_hit_rate:<8.2%} "
            f"{agg.mean_latency_ms:<12.2f}"
        )

    # Ranking
    print("\n" + "=" * 90)
    print("RANKING BY F1@10 SCORE")
    print("=" * 90)
    ranked = sorted(aggregate.items(), key=lambda x: x[1].mean_f1, reverse=True)
    for rank, (algo_name, agg) in enumerate(ranked, 1):
        marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"  {rank}."
        print(
            f"{marker} {algo_name:<16} "
            f"F1={agg.mean_f1:.4f} P={agg.mean_precision:.3f} R={agg.mean_recall:.3f} "
            f"MRR={agg.mean_mrr:.3f} NDCG={agg.mean_ndcg:.3f} Hit={agg.mean_hit_rate:.1%} "
            f"({agg.mean_latency_ms:.1f}ms)"
        )

    # Save results
    output_data = {
        "meta": {
            "source": "Real MuSiQue multi-hop dataset",
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "questions_evaluated": len(questions),
            "top_k": args.top_k,
            "algorithms": [algo_name for algo_name, _ in algorithms],
        },
        "per_question": {
            algo_name: [asdict(m) for m in metrics_list]
            for algo_name, metrics_list in all_results.items()
        },
        "aggregate": {algo_name: asdict(agg) for algo_name, agg in aggregate.items()},
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n✓ Detailed results saved to {output_path}")

    # Summary CSV for quick reference
    csv_stem = output_path.stem.replace("results", "summary")
    csv_path = output_path.parent / (csv_stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("Algorithm,P@10,R@10,F1@10,MRR,NDCG@10,Hit_Rate,Mean_Latency_ms\n")
        for algo_name in [name for name, _ in algorithms]:
            agg = aggregate[algo_name]
            f.write(
                f"{algo_name},"
                f"{agg.mean_precision:.4f},"
                f"{agg.mean_recall:.4f},"
                f"{agg.mean_f1:.4f},"
                f"{agg.mean_mrr:.4f},"
                f"{agg.mean_ndcg:.4f},"
                f"{agg.mean_hit_rate:.4f},"
                f"{agg.mean_latency_ms:.2f}\n"
            )
    print(f"✓ Summary CSV saved to {csv_path}")

    print("\n" + "=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
