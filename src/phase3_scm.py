"""
Phase 3: Mixed Structural Causal Model (SCM)

Fits structural equations for each node in the causal DAG. The key technical
contribution is the MIXED model selection: different model classes are used
for different variable types, avoiding the global linearity assumption that
would violate the linear probability model constraint for binary variables.

Model selection:
  Binary variables       → LogisticRegression (correct probability model)
  Outcome variable       → GradientBoostingRegressor (captures non-linearity)
  Continuous variables   → LinearRegression (appropriate for linear structure)

Using LinearRegression for binary outcomes (the linear probability model)
is a known methodological error that produces probabilities outside [0,1]
and biased coefficients. This SCM explicitly avoids that error.

Reference for mixed SCM approach:
  Peters, Janzing, Schölkopf (2017). Elements of Causal Inference.
  MIT Press. Chapter 6.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Ground truth structural coefficients for the manufacturing domain.
# These are planted values from the data generator — used ONLY for validation
# in the coefficient recovery table, never displayed as computed results.
_MFG_GROUND_TRUTH = {
    ('supplier_a', 'material_lead_time'):         7.4,
    ('material_lead_time', 'shipment_delay'):      0.9,
    ('order_complexity', 'machine_queue_length'):  0.8,
    ('machine_queue_length', 'approval_duration'): 1.3,
    ('export_flag', 'approval_duration'):          2.0,
    ('carrier_express', 'shipment_delay'):         -0.6,
    ('order_complexity', 'shipment_delay'):        0.20,
    ('approval_duration', 'shipment_delay'):       0.35,
    ('order_complexity', 'supplier_a'):            float('nan'),  # non-linear
}


def _select_model(node: str, binary_vars: list, outcome_var: str):
    """
    Select the appropriate structural equation model for a given node.

    This is the core of the mixed SCM: binary nodes get logistic regression,
    the outcome gets gradient boosting, everything else gets linear regression.
    """
    if node in binary_vars:
        return LogisticRegression(max_iter=1000, random_state=42)
    elif node == outcome_var:
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
    else:
        return LinearRegression()


def _model_type_name(model) -> str:
    """Return a canonical string label for the model type."""
    class_name = type(model).__name__
    if 'Logistic' in class_name:
        return 'logistic'
    elif 'GradientBoosting' in class_name:
        return 'gradient_boosting'
    else:
        return 'linear'


def fit_scm(df: pd.DataFrame,
             dag: nx.DiGraph,
             binary_vars: list,
             outcome_var: str) -> dict:
    """
    Fit structural equations for all non-root nodes in the causal DAG.

    Root nodes (no parents) are exogenous — no equation is fitted for them.
    The returned dict maps each non-root node to its fitted model and metadata.

    Parameters
    ----------
    df : pd.DataFrame
    dag : nx.DiGraph — the causal DAG from Phase 2
    binary_vars : list of str — variable names to model with LogisticRegression
    outcome_var : str — variable name to model with GradientBoostingRegressor

    Returns
    -------
    dict mapping node → {model, parents, model_type, r2_score, metric_label}
    """
    scm = {}

    try:
        topo_order = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        logger.warning("[SCM] DAG has cycle — cannot fit SCM.")
        return scm

    for node in topo_order:
        if node not in df.columns:
            continue

        parents = list(dag.predecessors(node))
        parents = [p for p in parents if p in df.columns]

        if len(parents) == 0:
            continue  # Root / exogenous node

        model = _select_model(node, binary_vars, outcome_var)
        model_type = _model_type_name(model)

        # Fit on complete cases for this node and its parents
        cols = [node] + parents
        subset = df[cols].dropna()
        X = subset[parents].values
        y = subset[node].values

        try:
            model.fit(X, y)
        except Exception as e:
            logger.warning(f"[SCM] Failed to fit {node}: {e}")
            continue

        # Cross-validated score (5-fold, out-of-sample) — more honest than in-sample
        cv_model = _select_model(node, binary_vars, outcome_var)
        if model_type == 'logistic':
            try:
                cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='roc_auc')
                r2_score = float(np.mean(cv_scores))
            except Exception:
                try:
                    proba = model.predict_proba(X)[:, 1]
                    r2_score = roc_auc_score(y, proba)
                except Exception:
                    r2_score = 0.5
            metric_label = 'CV-AUC'
        elif model_type == 'gradient_boosting':
            # Use lighter GBM for CV to keep it fast; full model already fitted above
            cv_gbm = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
            try:
                cv_scores = cross_val_score(cv_gbm, X, y, cv=5, scoring='r2')
                r2_score = max(0.0, float(np.mean(cv_scores)))
            except Exception:
                r2_score = max(0.0, model.score(X, y))
            metric_label = 'CV-R²'
        else:
            try:
                cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='r2')
                r2_score = max(0.0, float(np.mean(cv_scores)))
            except Exception:
                r2_score = max(0.0, model.score(X, y))
            metric_label = 'CV-R²'

        scm[node] = {
            'model':        model,
            'parents':      parents,
            'model_type':   model_type,
            'r2_score':     r2_score,
            'metric_label': metric_label,
        }
        logger.debug(f"[SCM] {node}: {model_type} ({metric_label}={r2_score:.3f})")

    return scm


def get_coefficients(scm: dict, domain: str = 'manufacturing') -> pd.DataFrame:
    """
    Extract structural coefficients from the fitted SCM into a tidy DataFrame.

    For LinearRegression: coefficients directly.
    For GradientBoostingRegressor: normalised feature importances.
    For LogisticRegression: exp(coefficient) — odds ratios.

    Ground truth values are included for the manufacturing domain to enable
    coefficient recovery validation. Healthcare domain uses NaN for ground truth
    (structural demonstration only, not numerical validation).
    """
    rows = []
    gt = _MFG_GROUND_TRUTH if domain == 'manufacturing' else {}

    for node, eq in scm.items():
        model = eq['model']
        parents = eq['parents']
        model_type = eq['model_type']
        metric_label = eq['metric_label']
        metric_value = eq['r2_score']

        if model_type == 'linear':
            coefs = model.coef_
            for parent, coef in zip(parents, coefs):
                gt_val = gt.get((parent, node), float('nan'))
                rows.append({
                    'child':            node,
                    'parent':           parent,
                    'estimated_value':  round(coef, 4),
                    'ground_truth_value': gt_val,
                    'abs_error':        abs(coef - gt_val) if not np.isnan(gt_val) else float('nan'),
                    'model_type':       model_type,
                    'metric_label':     metric_label,
                    'metric_value':     round(metric_value, 4),
                })

        elif model_type == 'gradient_boosting':
            importances = model.feature_importances_
            total = importances.sum()
            norm_importances = importances / total if total > 0 else importances
            for parent, imp in zip(parents, norm_importances):
                gt_val = gt.get((parent, node), float('nan'))
                rows.append({
                    'child':             node,
                    'parent':            parent,
                    'estimated_value':   round(imp, 4),
                    'ground_truth_value': gt_val,
                    'abs_error':         abs(imp - gt_val) if not np.isnan(gt_val) else float('nan'),
                    'model_type':        model_type,
                    'metric_label':      metric_label,
                    'metric_value':      round(metric_value, 4),
                })

        elif model_type == 'logistic':
            coefs = model.coef_[0]
            odds_ratios = np.exp(coefs)
            for parent, odds in zip(parents, odds_ratios):
                gt_val = gt.get((parent, node), float('nan'))
                rows.append({
                    'child':             node,
                    'parent':            parent,
                    'estimated_value':   round(odds, 4),
                    'ground_truth_value': gt_val,
                    'abs_error':         float('nan'),  # odds ratios vs linear gt are not comparable
                    'model_type':        model_type,
                    'metric_label':      metric_label,
                    'metric_value':      round(metric_value, 4),
                })

    df_coefs = pd.DataFrame(rows)
    if not df_coefs.empty:
        df_coefs = df_coefs.sort_values('abs_error', ascending=False, na_position='last')
    return df_coefs


def predict_outcome(scm: dict,
                    df: pd.DataFrame,
                    dag: nx.DiGraph,
                    intervention: dict = None) -> np.ndarray:
    """
    Predict outcome under optional do-intervention.

    Propagates through the SCM in topological order. An intervention
    do(T=t) fixes variable T to value t, overriding its structural equation.

    For LogisticRegression nodes: predict_proba()[:,1] is used as the node
    value in downstream equations (probability as a continuous predictor).
    """
    if intervention is None:
        intervention = {}

    try:
        topo_order = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        return np.zeros(len(df))

    values = df.copy()

    for node in topo_order:
        if node in intervention:
            values[node] = float(intervention[node])
        elif node in scm:
            eq = scm[node]
            parents = eq['parents']
            available_parents = [p for p in parents if p in values.columns]
            if not available_parents:
                continue
            X = values[available_parents].values
            model = eq['model']
            model_type = eq['model_type']

            try:
                if model_type == 'logistic':
                    values[node] = model.predict_proba(X)[:, 1]
                else:
                    values[node] = model.predict(X)
            except Exception as e:
                logger.warning(f"[SCM] Prediction failed for {node}: {e}")

    outcome_var = [k for k in scm if k in values.columns]
    # Return outcome column if identifiable, else zeros
    for node in reversed(topo_order):
        if node in scm and node in values.columns:
            return values[node].values

    return np.zeros(len(df))


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.generate_data import (load_or_generate, BINARY_VARS, OUTCOME_VAR,
                                     NUMERIC_VARS, GROUND_TRUTH_EDGES)
    from src.phase2_discovery import discover_dag

    df = load_or_generate()
    dag = discover_dag(df, NUMERIC_VARS, GROUND_TRUTH_EDGES, OUTCOME_VAR)
    scm = fit_scm(df, dag, BINARY_VARS, OUTCOME_VAR)
    coefs = get_coefficients(scm)

    print("[Phase 3 — Mixed SCM]")
    print(f"  Models fitted: {len(scm)}")
    for node, eq in scm.items():
        print(f"  {node}: {eq['model_type']} "
              f"({eq['metric_label']}={eq['r2_score']:.3f})")
    print("Phase 3 complete.")
