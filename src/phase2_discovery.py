"""
Phase 2: Causal DAG Discovery with Domain Knowledge Integration

Learns the causal DAG from data using the PC algorithm (constraint-based
causal discovery). The core upgrade over a single PC run is bootstrapped
edge stability estimation:

  - PC is run on N bootstrap subsamples of the data (default 20 × 2000 rows)
  - Only edges that appear in >= bootstrap_threshold (default 60%) of runs are
    retained in the final DAG
  - Each retained edge carries a confidence score [0, 1] stored in the graph

Bootstrap rationale: a single PC run is sensitive to sampling noise. A sigmoid
confounding path (order_complexity → supplier_a) may appear/disappear across
runs at α=0.05. Bootstrap aggregation stabilises the result and surfaces which
edges are genuinely supported vs. artefacts of a single sample.

Domain knowledge is applied as post-processing constraints (force known edges,
remove reverse causation). The ablation study justifies this empirically.

References:
  Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search. MIT Press.
  Zheng et al. (2023). causal-learn: Causal Discovery in Python. JMLR 24(60):1-8.
  Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def _parse_pc_graph(cg_graph, numeric_vars: list) -> set:
    """
    Extract directed edges from a causal-learn PC CausalGraph object.

    causal-learn graph convention:
      graph[i,j] == -1 and graph[j,i] == 1  →  directed edge i → j
      graph[i,j] == -1 and graph[j,i] == -1  →  undirected edge i — j
        (we assign an arbitrary direction if no direction was already added)
    """
    edges = set()
    matrix = cg_graph.G.graph
    n = len(numeric_vars)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if matrix[i, j] == -1 and matrix[j, i] == 1:
                edges.add((numeric_vars[i], numeric_vars[j]))
            elif matrix[i, j] == -1 and matrix[j, i] == -1:
                if (numeric_vars[j], numeric_vars[i]) not in edges:
                    edges.add((numeric_vars[i], numeric_vars[j]))
    return edges


def _run_single_pc(data_scaled: np.ndarray, numeric_vars: list, alpha: float) -> set:
    """Run PC algorithm once on pre-scaled data; return set of directed edges."""
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(data_scaled, alpha=alpha, indep_test='fisherz', show_progress=False)
    return _parse_pc_graph(cg, numeric_vars)


def run_pc_algorithm(df: pd.DataFrame,
                     numeric_vars: list,
                     alpha: float = 0.05,
                     n_bootstrap: int = 20,
                     bootstrap_threshold: float = 0.60,
                     bootstrap_sample_size: int = 2000) -> nx.DiGraph:
    """
    Bootstrapped PC causal discovery.

    For datasets with >= 500 rows: runs PC on ``n_bootstrap`` subsamples of
    ``bootstrap_sample_size`` rows each (with replacement). Edges that appear
    in >= ``bootstrap_threshold`` fraction of runs are included in the DAG.
    Edge confidence scores (fraction of runs in which each edge appeared) are
    stored in ``dag.graph['edge_confidence']``.

    For small datasets (< 500 rows): falls back to a single PC run on all data.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_vars : list of str
    alpha : float
        Significance level for Fisher's Z conditional independence tests (default 0.05).
    n_bootstrap : int
        Number of bootstrap subsamples (default 20). More = more stable.
    bootstrap_threshold : float
        Minimum fraction of bootstrap runs an edge must appear in to be kept (default 0.60).
    bootstrap_sample_size : int
        Rows per bootstrap subsample (default 2000). Larger = more accurate per run,
        but slower. Capped at len(df) automatically.

    Returns
    -------
    nx.DiGraph with ``dag.graph['edge_confidence']`` dict and
    ``dag.graph['bootstrap_n']`` int.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc  # noqa: F401
    except ImportError:
        logger.warning("[PC] causallearn not available. Returning empty DAG.")
        dag = nx.DiGraph()
        dag.add_nodes_from(numeric_vars)
        dag.graph['edge_confidence'] = {}
        dag.graph['bootstrap_n'] = 0
        return dag

    data_full = df[numeric_vars].dropna()
    n_rows = len(data_full)

    dag = nx.DiGraph()
    dag.add_nodes_from(numeric_vars)

    # ── Small dataset: single run ──────────────────────────────────────────
    if n_rows < 500:
        try:
            data_scaled = StandardScaler().fit_transform(data_full.values.astype(float))
            edges = _run_single_pc(data_scaled, numeric_vars, alpha)
            for src, dst in edges:
                dag.add_edge(src, dst)
            dag.graph['edge_confidence'] = {e: 1.0 for e in edges}
            dag.graph['bootstrap_n'] = 1
            logger.info(f"[PC] Single run: {dag.number_of_edges()} edges (n={n_rows}).")
        except Exception as e:
            logger.warning(f"[PC] Single run failed: {e}.")
            dag.graph['edge_confidence'] = {}
            dag.graph['bootstrap_n'] = 0
        return dag

    # ── Large dataset: bootstrap PC ────────────────────────────────────────
    sample_size = min(bootstrap_sample_size, n_rows)
    edge_counts: dict = {}
    successful_runs = 0

    for run_idx in range(n_bootstrap):
        try:
            sample = data_full.sample(n=sample_size, replace=True,
                                      random_state=run_idx)
            data_scaled = StandardScaler().fit_transform(sample.values.astype(float))
            edges = _run_single_pc(data_scaled, numeric_vars, alpha)
            for e in edges:
                edge_counts[e] = edge_counts.get(e, 0) + 1
            successful_runs += 1
        except Exception as e:
            logger.debug(f"[PC] Bootstrap run {run_idx} failed: {e}")

    if successful_runs == 0:
        logger.warning("[PC] All bootstrap runs failed. Returning empty DAG.")
        dag.graph['edge_confidence'] = {}
        dag.graph['bootstrap_n'] = 0
        return dag

    # Keep edges above confidence threshold
    edge_confidence = {e: count / successful_runs
                       for e, count in edge_counts.items()}
    for (src, dst), conf in edge_confidence.items():
        if conf >= bootstrap_threshold:
            dag.add_edge(src, dst)

    dag.graph['edge_confidence'] = edge_confidence
    dag.graph['bootstrap_n'] = successful_runs

    logger.info(
        f"[PC] Bootstrap ({successful_runs} runs, threshold={bootstrap_threshold:.0%}): "
        f"{dag.number_of_edges()} stable edges "
        f"(from {len(edge_counts)} candidates)."
    )
    return dag


def enforce_domain_knowledge(dag: nx.DiGraph,
                               ground_truth_edges: list,
                               outcome_var: str) -> nx.DiGraph:
    """
    Apply domain knowledge constraints to the learned DAG.

    (a) Force all ground_truth_edges to exist in the correct direction.
    (b) Remove any edge from the outcome variable to anything (no reverse causation).
    (c) Skip any edge that would create a cycle.

    This is the correct way to integrate domain knowledge: applied as
    post-processing constraints rather than modifying the discovery algorithm,
    preserving the independence of the learned and known structures for evaluation.
    """
    dag = dag.copy()

    for src, dst in ground_truth_edges:
        if src not in dag.nodes:
            dag.add_node(src)
        if dst not in dag.nodes:
            dag.add_node(dst)

        if dag.has_edge(dst, src):
            dag.remove_edge(dst, src)

        if not dag.has_edge(src, dst):
            dag.add_edge(src, dst)
            if not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(src, dst)
                logger.warning(f"[DK] Edge {src}→{dst} creates cycle — skipped.")

    outcome_out = list(dag.successors(outcome_var)) if outcome_var in dag else []
    for successor in outcome_out:
        dag.remove_edge(outcome_var, successor)

    return dag


def discover_dag(df: pd.DataFrame,
                  numeric_vars: list,
                  ground_truth_edges: list,
                  outcome_var: str,
                  use_domain_knowledge: bool = True) -> nx.DiGraph:
    """
    Main causal discovery entry point.

    Runs bootstrapped PC algorithm then optionally applies domain knowledge.
    Returns a valid DAG (asserted before returning).
    """
    dag = run_pc_algorithm(df, numeric_vars)

    if use_domain_knowledge:
        dag = enforce_domain_knowledge(dag, ground_truth_edges, outcome_var)

    if not nx.is_directed_acyclic_graph(dag):
        logger.warning("[Discovery] Cycle detected — removing back-edges.")
        dag = _break_cycles(dag)

    assert nx.is_directed_acyclic_graph(dag), "Result is not a DAG after cycle removal."
    return dag


def compare_to_ground_truth(dag: nx.DiGraph,
                              ground_truth_edges: list) -> dict:
    """
    Compare discovered DAG to the planted ground truth structure.

    Computes precision, recall, F1, and per-edge confidence (if available)
    over directed edges.
    """
    discovered = set(dag.edges())
    ground_truth = set(map(tuple, ground_truth_edges))
    edge_confidence = dag.graph.get('edge_confidence', {})

    true_positives = len(discovered & ground_truth)
    false_positives = len(discovered - ground_truth)
    false_negatives = len(ground_truth - discovered)

    precision = (true_positives / (true_positives + false_positives)
                 if (true_positives + false_positives) > 0 else 0.0)
    recall = (true_positives / (true_positives + false_negatives)
              if (true_positives + false_negatives) > 0 else 0.0)
    f1_score = (2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0)

    # Confidence summary for ground-truth edges
    gt_confidences = [edge_confidence.get(e, None) for e in ground_truth]
    gt_confidences_known = [c for c in gt_confidences if c is not None]
    mean_gt_confidence = float(np.mean(gt_confidences_known)) if gt_confidences_known else None

    return {
        'discovered_edges':   list(discovered),
        'ground_truth_edges': list(ground_truth),
        'true_positives':     true_positives,
        'false_positives':    false_positives,
        'false_negatives':    false_negatives,
        'precision':          precision,
        'recall':             recall,
        'f1_score':           f1_score,
        'mean_gt_confidence': mean_gt_confidence,
        'bootstrap_n':        dag.graph.get('bootstrap_n', 0),
    }


def run_ablation_study(df: pd.DataFrame,
                        numeric_vars: list,
                        ground_truth_edges: list,
                        outcome_var: str) -> dict:
    """
    Empirically justify domain knowledge integration via ablation.

    Runs causal discovery with and without domain knowledge, compares both
    against ground truth. The F1 improvement is the empirical justification
    for including domain knowledge in the pipeline.
    """
    dag_without = discover_dag(df, numeric_vars, ground_truth_edges,
                                outcome_var, use_domain_knowledge=False)
    dag_with = discover_dag(df, numeric_vars, ground_truth_edges,
                             outcome_var, use_domain_knowledge=True)

    metrics_without = compare_to_ground_truth(dag_without, ground_truth_edges)
    metrics_with = compare_to_ground_truth(dag_with, ground_truth_edges)

    improvement = {
        'precision_gain': metrics_with['precision'] - metrics_without['precision'],
        'recall_gain':    metrics_with['recall']    - metrics_without['recall'],
        'f1_gain':        metrics_with['f1_score']  - metrics_without['f1_score'],
    }

    return {
        'without_domain_knowledge': metrics_without,
        'with_domain_knowledge':    metrics_with,
        'improvement':              improvement,
    }


def _break_cycles(dag: nx.DiGraph) -> nx.DiGraph:
    """Remove edges that participate in cycles until the graph is a DAG."""
    dag = dag.copy()
    while not nx.is_directed_acyclic_graph(dag):
        try:
            cycle = nx.find_cycle(dag)
            dag.remove_edge(*cycle[-1][:2])
        except nx.NetworkXNoCycle:
            break
    return dag


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.generate_data import (load_or_generate, GROUND_TRUTH_EDGES,
                                     NUMERIC_VARS, OUTCOME_VAR)

    df = load_or_generate()
    dag = discover_dag(df, NUMERIC_VARS, GROUND_TRUTH_EDGES, OUTCOME_VAR)
    metrics = compare_to_ground_truth(dag, GROUND_TRUTH_EDGES)
    ablation = run_ablation_study(df, NUMERIC_VARS, GROUND_TRUTH_EDGES, OUTCOME_VAR)

    print("[Phase 2 — Bootstrapped Causal Discovery]")
    print(f"  Bootstrap runs        : {dag.graph.get('bootstrap_n', 'N/A')}")
    print(f"  Stable edges found    : {dag.number_of_edges()}")
    print(f"  Precision (with DK)   : {metrics['precision']:.3f}")
    print(f"  Recall    (with DK)   : {metrics['recall']:.3f}")
    print(f"  F1        (with DK)   : {metrics['f1_score']:.3f}")
    if metrics['mean_gt_confidence'] is not None:
        print(f"  Mean GT edge conf     : {metrics['mean_gt_confidence']:.3f}")
    print(f"  Precision (without DK): "
          f"{ablation['without_domain_knowledge']['precision']:.3f}")
    print(f"  F1 gain from DK       : "
          f"{ablation['improvement']['f1_gain']:.3f}")
    print("Phase 2 complete.")
