"""
GraphRAG: Knowledge Graph-Augmented Retrieval for Research Papers
=================================================================
Novel contribution: dual-edge knowledge graph combining two complementary
edge types for richer structural connectivity:

  1. Vocabulary-Jaccard edges  (semantic similarity)
     Computed from Jaccard similarity of top-200 TF terms per paper.
     Threshold: Jaccard >= similarity_threshold (default 0.10).
     Captures TOPICAL proximity between papers.

  2. Citation co-occurrence edges  (bibliographic coupling)
     Extracted from reference sections of raw text files using regex.
     Matched against paper corpus via title substring overlap.
     Captures DIRECT REFERENCE relationships between papers.

The two edge types have distinct weights and are fused additively:
  combined_weight = jaccard_weight + citation_weight * citation_boost

This dual-edge graph is a novel contribution over single-signal graphs.
=================================================================

Architecture:
  1. Build paper-level knowledge graph from chunk metadata
       - Nodes  : papers (identified by filename)
       - Edges  : weighted Jaccard similarity of top-200 vocabulary terms
                  (approximates semantic relatedness / co-citation proxy)
  2. Graph-augmented retrieval pipeline:
       a. Hybrid dense (FAISS) + sparse (BM25) retrieval  ->  initial candidates
       b. Map retrieved chunks -> paper nodes in graph
       c. 1-hop graph expansion: fetch neighbour papers by edge weight
       d. Retrieve best BM25 chunk from each neighbour paper
       e. RRF re-ranking of full candidate set (dense + BM25 + graph)
  3. LLM answer generation with source-attributed context

Comparison baseline identifiers (for eval_compare.py):
  - 'dense'        : FAISS cosine only           (OllamaRAG path)
  - 'hybrid'       : BM25 + FAISS + RRF          (EnhancedRAG)
  - 'graph'        : FAISS + graph expansion      (this file, use_bm25=False)
  - 'hybrid_graph' : BM25 + FAISS + graph + RRF  (this file, use_bm25=True)
"""

import os
import pickle
import re
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, Counter

import numpy as np
import faiss
import requests
import networkx as nx

logger = logging.getLogger(__name__)

from nim_embedder import embed_text as _nim_embed_text, NIM_EMBED_DIM, NIM_EMBED_MODEL

# ---------------------------------------------------------------------------
# Stopwords (lightweight, no NLTK dependency)
# ---------------------------------------------------------------------------
_STOP_WORDS: Set[str] = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'we', 'our', 'they', 'their', 'he',
    'she', 'his', 'her', 'i', 'my', 'you', 'your', 'not', 'no', 'as',
    'if', 'than', 'then', 'so', 'also', 'which', 'who', 'when', 'where',
    'what', 'how', 'paper', 'model', 'method', 'approach', 'system',
    'using', 'used', 'use', 'show', 'shown', 'based', 'propose',
    'proposed', 'present', 'presented', 'result', 'results', 'work',
}


class GraphRAG:
    """
    Knowledge Graph-Augmented RAG system.

    Drop-in replacement for EnhancedRAG in evaluations:
      - same .load(index_dir)
      - same .search(query, top_k)  ->  List[Dict]
      - same .query(question, top_k, verbose) -> (answer, sources)

    Extra capability:
      - .get_graph_stats()  -> dict  (for paper tables)
    """

    def __init__(
        self,
        ollama_embed: str = "bge-m3",
        ollama_llm: str = "deepseek-r1:7b",
        ollama_url: str = "http://localhost:11434",
        use_bm25: bool = True,
        similarity_threshold: float = 0.30,
        max_graph_hops: int = 2,
        max_expansion_papers: int = 3,
        use_adaptive_budget: bool = True,
        expansion_mode: str = "qbpr",
    ):
        """
        Args:
            ollama_embed          : Ollama embedding model name
            ollama_llm            : Ollama generation model name
            ollama_url            : Ollama server URL
            use_bm25              : Enable BM25 hybrid retrieval
            similarity_threshold  : Minimum Jaccard similarity to add a graph edge
            max_graph_hops        : Number of hops for graph expansion (1 or 2)
            max_expansion_papers  : Max neighbour papers to expand per query
            use_adaptive_budget   : Use data-driven graph slot allocation instead
                                    of fixed k//4 (default True)
            expansion_mode        : Graph traversal mode - '1hop', 'ppr', or
                                    'qbpr' (query-biased personalized pagerank)
        """
        self.ollama_embed = ollama_embed
        self.ollama_llm = ollama_llm
        self.ollama_url = ollama_url
        self.use_bm25 = use_bm25
        self.similarity_threshold = similarity_threshold
        self.max_graph_hops = max_graph_hops
        self.max_expansion_papers = max_expansion_papers
        self.use_adaptive_budget = use_adaptive_budget
        self.expansion_mode = expansion_mode

        # FAISS index
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []

        # BM25 state
        self.bm25_docs: Optional[List[List[str]]] = None
        self.idf: Optional[Dict[str, float]] = None
        self.avgdl: float = 1.0

        # Knowledge graph
        self.graph: nx.Graph = nx.Graph()
        self.paper_chunks: Dict[str, List[int]] = defaultdict(list)
        self.paper_vocab: Dict[str, Set[str]] = {}
        # Maps paper_id -> short title fingerprint (for citation matching)
        self.paper_title_tokens: Dict[str, Set[str]] = {}
        # Path to raw text directory for citation extraction
        self._raw_text_dir: Optional[Path] = None

        # Cross-encoder reranker (lazy-loaded on first search)
        self._reranker = None


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_reranker(self):
        """Lazy-load the NIM Mistral-4B reranker (cached after first call)."""
        if self._reranker is None:
            from core.nim_reranker import NIMReranker
            self._reranker = NIMReranker()
        return self._reranker

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text, strip stop-words, keep meaningful alpha-numeric tokens."""
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', text.lower())
        return [t for t in tokens if t not in _STOP_WORDS]

    def get_embedding(self, text: str, input_type: str = "query") -> np.ndarray:
        """Embed via NIM (model: NIM_EMBED_MODEL). See nim_embedder for details."""
        return _nim_embed_text(text, input_type=input_type)

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _build_bm25(self):
        """Build BM25 inverted index from loaded chunks."""
        logger.info("Building BM25 index ...")
        self.bm25_docs = [self._tokenize(c) for c in self.chunks]
        df: Counter = Counter()
        for doc in self.bm25_docs:
            df.update(set(doc))
        N = len(self.bm25_docs)
        self.idf = {
            t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            for t in df
        }
        self.avgdl = (
            sum(len(d) for d in self.bm25_docs) / N if N > 0 else 1.0
        )
        logger.info(f"BM25 ready: {N} docs, vocab={len(self.idf)}")

    def _bm25_score(
        self, query_tokens: List[str], doc_idx: int,
        k1: float = 1.5, b: float = 0.75
    ) -> float:
        """BM25 relevance score for a single document."""
        if not self.bm25_docs or not self.idf:
            return 0.0
        doc = self.bm25_docs[doc_idx]
        doc_len = len(doc)
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = doc.count(term)
            score += self.idf[term] * (tf * (k1 + 1)) / (
                tf + k1 * (1.0 - b + b * doc_len / self.avgdl)
            )
        return score

    # ------------------------------------------------------------------
    # Knowledge graph construction
    # ------------------------------------------------------------------

    def build_knowledge_graph(self, raw_text_dir: Optional[str] = None):
        """
        Build dual-edge paper-level knowledge graph.

        Edge type 1 - Vocabulary-Jaccard (semantic similarity):
          Node pairs with Jaccard similarity >= threshold on their
          top-200 TF vocabulary terms receive a weighted edge.

        Edge type 2 - Citation co-occurrence (bibliographic coupling):
          Reference sections of raw .txt files are scanned for paper
          titles already in the corpus. Each cross-citation adds a
          citation-weighted edge (weight = 1.5 x Jaccard normalisation).

        Combined edge weight:
          w = jaccard_weight + citation_matches * 0.15
        """
        logger.info("Building dual-edge knowledge graph ...")

        if raw_text_dir:
            self._raw_text_dir = Path(raw_text_dir)

        # Map chunks -> papers and build vocabulary
        self.paper_chunks.clear()
        self.paper_vocab.clear()
        self.paper_title_tokens.clear()

        for idx, meta in enumerate(self.metadata):
            paper_id = meta["file"]
            self.paper_chunks[paper_id].append(idx)

        for paper_id, chunk_indices in self.paper_chunks.items():
            vocab: Counter = Counter()
            for idx in chunk_indices:
                vocab.update(self._tokenize(self.chunks[idx]))
            self.paper_vocab[paper_id] = {
                t for t, _ in vocab.most_common(200)
            }
            # Title fingerprint: first 15 significant tokens of paper_id (filename)
            name_tokens = set(self._tokenize(paper_id.replace('_', ' ')))
            self.paper_title_tokens[paper_id] = name_tokens
            self.graph.add_node(paper_id, chunk_count=len(chunk_indices))

        papers = list(self.paper_chunks.keys())

        # -- Edge type 1: Vocabulary-Jaccard ----------------------------
        jaccard_edges = 0
        edge_weights: Dict[Tuple[str, str], float] = {}
        edge_citation_weight: Dict[Tuple[str, str], float] = defaultdict(float)
        edge_citation_overlap: Dict[Tuple[str, str], int] = defaultdict(int)
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                p1, p2 = papers[i], papers[j]
                v1, v2 = self.paper_vocab[p1], self.paper_vocab[p2]
                if not v1 or not v2:
                    continue
                jaccard = len(v1 & v2) / len(v1 | v2)
                if jaccard >= self.similarity_threshold:
                    key = (min(p1, p2), max(p1, p2))
                    edge_weights[key] = jaccard
                    jaccard_edges += 1

        # -- Edge type 2: Citation co-occurrence -------------------------
        citation_edges = 0
        if self._raw_text_dir and self._raw_text_dir.is_dir():
            citation_edges = self._extract_citation_edges(
                papers,
                edge_weights,
                edge_citation_weight,
                edge_citation_overlap,
            )

        # Write final edges to graph
        for (p1, p2), weight in edge_weights.items():
            citation_weight = float(edge_citation_weight.get((p1, p2), 0.0))
            citation_overlap = int(edge_citation_overlap.get((p1, p2), 0))
            self.graph.add_edge(
                p1,
                p2,
                weight=round(weight, 4),
                citation_weight=round(citation_weight, 4),
                citation_overlap=citation_overlap,
                has_citation_edge=citation_weight > 0.0,
            )

        stats = self.get_graph_stats()
        logger.info(
            f"Graph: {stats['nodes']} nodes, {stats['edges']} edges "
            f"(Jaccard={jaccard_edges}, Citation={citation_edges})"
        )
        print(
            f"[Graph] Dual-edge: {stats['nodes']} papers, {stats['edges']} edges "
            f"[Jaccard={jaccard_edges} + Citation={citation_edges}]"
        )
        return self

    def _extract_reference_signatures(self, text: str) -> Set[str]:
        """Extract compact citation signatures from a paper's references/body text.

        Supported signatures:
          - DOI signatures: doi:10.xxxx/...
          - Author-year signatures: ay:surname:2023
        """
        signatures: Set[str] = set()

        # Focus on likely reference-rich segment first, fallback to full text.
        lower = text.lower()
        ref_idx = lower.find("references")
        segment = text[ref_idx:] if ref_idx >= 0 else text[-len(text) // 3 :]

        # DOI pattern is highly precise and can be shared across papers.
        for doi in re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", segment):
            signatures.add(f"doi:{doi.lower().rstrip('.,;)')}")

        # Author-year patterns (APA-like and narrative variants)
        ay_patterns = [
            re.compile(r"\b([A-Z][a-zA-Z\-]{2,})\s+et\s+al\.?\s*\(?((?:19|20)\d{2})[a-z]?\)?"),
            re.compile(r"\b([A-Z][a-zA-Z\-]{2,})\s*[,\(]\s*((?:19|20)\d{2})[a-z]?\)?"),
        ]
        for pattern in ay_patterns:
            for match in pattern.finditer(segment):
                surname = match.group(1).lower()
                year = match.group(2)
                signatures.add(f"ay:{surname}:{year}")

        return signatures

    def _extract_citation_edges(
        self,
        papers: List[str],
        edge_weights: Dict,
        edge_citation_weight: Dict,
        edge_citation_overlap: Dict,
        citation_boost: float = 0.15,
        min_shared_refs: int = 1,
    ) -> int:
        """Add citation-derived edges via shared reference signatures.

        This activates bibliographic coupling even when direct title-to-title
        matching is sparse in extracted raw text.

        Returns number of paper-pairs with non-zero citation overlap.
        """
        if not self._raw_text_dir:
            return 0

        paper_refs: Dict[str, Set[str]] = {}

        for paper_id in papers:
            txt_path = self._raw_text_dir / paper_id
            if not txt_path.exists():
                stem = paper_id.replace('.txt', '')
                txt_path = self._raw_text_dir / (stem + '.txt')
            if not txt_path.exists():
                paper_refs[paper_id] = set()
                continue

            try:
                text = txt_path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                paper_refs[paper_id] = set()
                continue

            paper_refs[paper_id] = self._extract_reference_signatures(text)

        citation_pairs = 0
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                p1, p2 = papers[i], papers[j]
                refs1 = paper_refs.get(p1, set())
                refs2 = paper_refs.get(p2, set())
                if not refs1 or not refs2:
                    continue

                overlap = len(refs1 & refs2)
                if overlap < min_shared_refs:
                    continue

                key = (min(p1, p2), max(p1, p2))
                # Cap overlap contribution to avoid swamping semantic component.
                citation_weight = citation_boost * float(min(overlap, 3))
                edge_weights[key] = edge_weights.get(key, 0.0) + citation_weight
                edge_citation_weight[key] = edge_citation_weight.get(key, 0.0) + citation_weight
                edge_citation_overlap[key] = max(edge_citation_overlap.get(key, 0), overlap)
                citation_pairs += 1

        return citation_pairs

    def _get_graph_neighbours(
        self, paper_ids: Set[str], query_emb: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        """
        Return top neighbour papers for the given set of seed papers,
        sorted by aggregated edge weight descending.
        """
        if self.expansion_mode in ("ppr", "qbpr"):
            try:
                import networkx as nx
                if len(self.graph) == 0:
                    return []
                if self.expansion_mode == "ppr":
                    n_seeds = len([s for s in paper_ids if s in self.graph])
                    if n_seeds == 0:
                        return []
                    personalisation = {
                        nid: (1.0 / n_seeds if nid in paper_ids else 0.0)
                        for nid in self.graph.nodes()
                    }
                else:  # qbpr
                    # Reconstruct paper embeddings from chunks and compute cosine against query
                    paper_sims = {}
                    for nid in self.graph.nodes():
                        if nid not in self.paper_chunks or not self.paper_chunks[nid]:
                            paper_sims[nid] = 0.0
                            continue
                        vecs = [self.index.reconstruct(i) for i in self.paper_chunks[nid][:50]] # limits up to 50 chunks for speed
                        avg_vec = np.mean(vecs, axis=0)
                        norm = np.linalg.norm(avg_vec)
                        if norm > 0:
                            avg_vec = avg_vec / norm
                        sim = float(np.dot(query_emb[0], avg_vec)) if query_emb is not None else 0.0
                        paper_sims[nid] = max(0.0, sim)
                    
                    total = sum(paper_sims.values()) or 1.0
                    personalisation = {nid: paper_sims.get(nid, 0.0) / total for nid in self.graph.nodes()}
                
                pr = nx.pagerank(self.graph, alpha=0.85, personalization=personalisation, weight="weight")
                candidates = [(nid, score) for nid, score in pr.items() if nid not in paper_ids]
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[: self.max_expansion_papers]
            except Exception as e:
                logger.warning(f"Failed to run {self.expansion_mode}: {e}. Falling back to 1hop.")

        scores: Dict[str, float] = defaultdict(float)
        seen = set(paper_ids)

        for paper in paper_ids:
            if paper not in self.graph:
                continue
            for hop1, data1 in self.graph[paper].items():
                if hop1 in seen:
                    scores[hop1] += data1.get("weight", 0.0)
                else:
                    scores[hop1] += data1.get("weight", 0.0)
                    seen.add(hop1)

                if self.max_graph_hops >= 2:
                    for hop2, data2 in self.graph[hop1].items():
                        if hop2 not in paper_ids:
                            scores[hop2] += data2.get("weight", 0.0) * 0.5

        # Exclude seed papers themselves
        candidates = [
            (p, s) for p, s in scores.items() if p not in paper_ids
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[: self.max_expansion_papers]

    def _best_chunk_from_paper(
        self, paper_id: str, query_tokens: List[str]
    ) -> Optional[Dict]:
        """Return highest-BM25 chunk from a given paper."""
        indices = self.paper_chunks.get(paper_id, [])
        if not indices:
            return None

        if self.bm25_docs:
            best_idx = max(
                indices, key=lambda i: self._bm25_score(query_tokens, i)
            )
            bm25_score = self._bm25_score(query_tokens, best_idx)
        else:
            best_idx = indices[0]
            bm25_score = 0.0

        # Use RRF-equivalent display score (1/(60+1)) so graph-expanded
        # chunks appear on the same scale as RRF-fused results (~0.016),
        # rather than showing the raw BM25 score which is a different scale.
        display_score = 1.0 / (60 + 1)

        return {
            "score": display_score,
            "text": self.chunks[best_idx],
            "metadata": self.metadata[best_idx],
            "scores": {
                "dense_score": 0.0,
                "bm25_score": bm25_score,
                "rrf_score": display_score,
                "graph_expanded": True,
            },
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Graph-augmented hybrid search.

        Pipeline:
          1. Dense FAISS retrieval   (always)
          2. BM25 retrieval          (if use_bm25=True)
          3. RRF fusion of 1+2
          4. Map top-K chunks -> paper nodes
          5. 1-hop graph expansion
          6. Append best BM25 chunk from each expanded paper
          7. Return top-K results (vector results first, graph expansions appended)
        """
        if self.index is None:
            raise RuntimeError("GraphRAG index not loaded. Call .load() first.")

        query_tokens = self._tokenize(query)
        from core.fusion import confidence_weighted_rrf

        # -- Compute graph slot budget -----------------------------------
        if self.use_adaptive_budget:
            try:
                from core.adaptive_budget import compute_graph_budget
                vocab_set = set()
                if self.bm25_docs:
                    for doc_tokens in self.bm25_docs:
                        vocab_set.update(doc_tokens)
                # Use placeholder lists; refined after retrieval below
                graph_slots = compute_graph_budget(
                    query, [], [], top_k, vocabulary=vocab_set or None,
                )
                graph_slots = max(1, min(graph_slots, self.max_expansion_papers))
            except ImportError:
                graph_slots = max(1, min(self.max_expansion_papers, top_k // 4))
        else:
            graph_slots = max(1, min(self.max_expansion_papers, top_k // 4))
        vector_slots = top_k - graph_slots

        # -- 1. Dense retrieval ------------------------------------------
        query_emb = self.get_embedding(query).reshape(1, -1)
        dense_scores, dense_indices = self.index.search(query_emb, vector_slots * 2)

        dense_ranked: List[Tuple[str, float]] = []
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            dense_ranked.append((str(int(idx)), float(score)))

        # -- 2. BM25 retrieval -------------------------------------------
        bm25_ranked: List[Tuple[str, float]] = []
        if self.use_bm25 and self.bm25_docs:
            bm25_pairs = sorted(
                [(i, self._bm25_score(query_tokens, i))
                 for i in range(len(self.chunks))],
                key=lambda x: x[1], reverse=True,
            )
            bm25_ranked = [
                (str(idx), score) for idx, score in bm25_pairs[: vector_slots * 2]
            ]

        # -- 3. Graph expansion (seeded from dense top results) ----------
        seed_papers: Set[str] = {
            self.metadata[int(idx)]["file"]
            for idx, _ in dense_ranked[:vector_slots]
        }
        neighbours = self._get_graph_neighbours(seed_papers, query_emb=query_emb)

        graph_ranked: List[Tuple[str, float]] = []
        graph_chunk_map: Dict[str, Dict] = {}   # str(idx) -> chunk dict
        for paper_id, edge_weight in neighbours[:graph_slots * 2]:
            chunk = self._best_chunk_from_paper(paper_id, query_tokens)
            if chunk:
                # Use a synthetic index key for graph chunks (negative to avoid
                # collision with real FAISS indices)
                key = f"g_{paper_id}"
                graph_ranked.append((key, edge_weight))
                graph_chunk_map[key] = chunk

        # -- 4. 3-channel CW-RRF fusion ---------------------------------
        ranked_lists = [dense_ranked]
        if bm25_ranked:
            ranked_lists.append(bm25_ranked)
        if graph_ranked:
            ranked_lists.append(graph_ranked)

        fused = confidence_weighted_rrf(ranked_lists)

        # -- 5. Slot reservation (AHR): guarantee graph slots in top-k --
        # Rfusion = top vector_slots from fused (may include graph results
        #           that ranked highly enough on their own)
        # RG      = top graph_slots from graph results NOT already in Rfusion
        rfusion_keys = [k for k, _ in fused[:vector_slots]]
        rfusion_set = set(rfusion_keys)

        rg_keys = [
            k for k, _ in fused
            if k in graph_chunk_map and k not in rfusion_set
        ][:graph_slots]

        final_keys = rfusion_keys + rg_keys

        # -- 6. Format results -------------------------------------------
        final: List[Dict] = []
        for key in final_keys:
            if key in graph_chunk_map:
                # Graph-expanded chunk - score from RRF position
                rrf_score = next((s for k, s in fused if k == key), 1.0 / 61)
                chunk = dict(graph_chunk_map[key])
                chunk["score"] = rrf_score
                chunk["scores"]["rrf_score"] = rrf_score
                final.append(chunk)
            else:
                idx = int(key)
                rrf_score = next((s for k, s in fused if k == key), 0.0)
                bm25_score = next(
                    (s for k, s in bm25_ranked if k == key), 0.0
                )
                dense_score = next(
                    (s for k, s in dense_ranked if k == key), 0.0
                )
                final.append({
                    "score": rrf_score,
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "scores": {
                        "dense_score": dense_score,
                        "bm25_score": bm25_score,
                        "rrf_score": rrf_score,
                        "graph_expanded": False,
                    },
                })

        # -- 7. Cross-encoder reranking (optional) ----------------------
        reranker = self._get_reranker()
        print(f"[DEBUG] reranker.available={reranker.available}, len(final)={len(final)}")
        if reranker.available and len(final) > 1:
            final = reranker.rerank(query, final, top_k=top_k, text_key="text")
            print(f"[DEBUG] reranked, first score={final[0].get('rerank_score', 'n/a')}")
            # Reassign scores based on new rank so the display reflects the
            # reranker's ordering (1/(60+rank)), not the original RRF scores.
            for rank, item in enumerate(final):
                item["score"] = 1.0 / (60 + rank + 1)
                item["scores"]["rrf_score"] = item["score"]

        return final[:top_k]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, question: str, context: str) -> str:
        """
        Generate answer. Tries Ollama first; falls back to OpenRouter.
        Set OPENROUTER_API_KEY environment variable to enable the fallback.
        """
        prompt = (
            "Based on the following research paper excerpts, "
            "answer the question concisely and accurately.\n\n"
            f"Context from papers:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        # Try Ollama first
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_llm, "prompt": prompt, "stream": False},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["response"]
        except Exception:
            pass

        # Fallback: OpenRouter — try models in order, skip on 429
        import os
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            _or_models = [
                "mistralai/mistral-7b-instruct:free",
                "qwen/qwen-2-7b-instruct:free",
                "google/gemma-3-4b-it:free",
            ]
            _last_err = ""
            for _model in _or_models:
                try:
                    resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}",
                                 "Content-Type": "application/json"},
                        json={
                            "model": _model,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
                    _last_err = f"{resp.status_code}: {resp.text[:120]}"
                    if resp.status_code in (429, 404):
                        continue  # try next model
                    return f"[OpenRouter error {_last_err}]"
                except Exception as e:
                    return f"[OpenRouter failed: {e}]"
            return f"[OpenRouter: all free models unavailable ({_last_err}). Retry shortly.]"

        return (
            "[Generation unavailable: Ollama is not running and OPENROUTER_API_KEY "
            "is not set. Set OPENROUTER_API_KEY in your .env file to enable "
            "cloud-based generation via OpenRouter.]"
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        verbose: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """
        Search and generate - drop-in replacement for EnhancedRAG.query().
        """
        sources = self.search(question, top_k=top_k)

        graph_count = sum(
            1 for r in sources if r["scores"].get("graph_expanded", False)
        )
        if verbose:
            print(
                f"Retrieved {len(sources)} chunks "
                f"({len(sources) - graph_count} vector, "
                f"{graph_count} graph-expanded)"
            )

        context_parts = []
        for i, r in enumerate(sources):
            tag = "[graph] " if r["scores"].get("graph_expanded") else ""
            context_parts.append(
                f"[Source {i+1}] {tag}{r['metadata']['file']}:\n{r['text']}"
            )
        context = "\n\n".join(context_parts)

        answer = self._generate(question, context)
        return answer, sources

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, index_dir: str) -> "GraphRAG":
        """
        Load FAISS index + chunk metadata, rebuild BM25 and knowledge graph.

        Compatible with indexes saved by OllamaRAG *or* EnhancedRAG.
        BM25 is always rebuilt from chunks (not assumed to be stored).
        """
        save_path = Path(index_dir)
        self.index = faiss.read_index(str(save_path / "faiss.index"))

        with open(save_path / "metadata.pkl", "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        self.metadata = data["metadata"]

        # BM25 may be pre-saved (EnhancedRAG) or needs rebuild (OllamaRAG)
        if data.get("bm25_docs") and self.use_bm25:
            self.bm25_docs = data["bm25_docs"]
            self.idf = data["idf"]
            self.avgdl = data.get("avgdl", 1.0)
            logger.info("BM25 loaded from index")
        elif self.use_bm25:
            self._build_bm25()

        print(
            f"[OK] GraphRAG loaded: {self.index.ntotal} vectors, "
            f"{len(self.chunks)} chunks from {len(set(m['file'] for m in self.metadata))} papers"
        )

        # Auto-detect raw text directory for citation extraction
        index_path = Path(index_dir)
        raw_text_candidates = [
            index_path.parent / "ingestion" / "raw_text",
            index_path.parent / "raw_text",
        ]
        raw_text_dir = next(
            (str(p) for p in raw_text_candidates if p.is_dir()), None
        )
        self.build_knowledge_graph(raw_text_dir=raw_text_dir)
        return self

    # ------------------------------------------------------------------
    # Stats (for paper tables)
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> Dict:
        """Return graph statistics suitable for a paper table."""
        if self.graph.number_of_nodes() == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "citation_edges": 0,
                "citation_edge_ratio": 0.0,
                "avg_degree": 0.0,
                "density": 0.0,
                "connected_components": 0,
            }

        degrees = [d for _, d in self.graph.degree()]
        total_edges = self.graph.number_of_edges()
        citation_edges = sum(
            1
            for _, _, attrs in self.graph.edges(data=True)
            if attrs.get("has_citation_edge", False)
        )
        citation_ratio = (citation_edges / total_edges) if total_edges > 0 else 0.0

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": total_edges,
            "citation_edges": citation_edges,
            "citation_edge_ratio": citation_ratio,
            "avg_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("GraphRAG smoke-test")
    print("=" * 60)

    rag = GraphRAG(use_bm25=True, similarity_threshold=0.10)
    rag.load("rag_index")

    stats = rag.get_graph_stats()
    print("\nGraph statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    answer, sources = rag.query(
        "What are the key advantages of hybrid retrieval in RAG systems?",
        top_k=5,
        verbose=True,
    )
    print(f"\nAnswer ({len(answer.split())} words):\n{answer[:400]}...")
    print(
        f"\nSources: {[s['metadata']['file'] for s in sources]}"
    )
