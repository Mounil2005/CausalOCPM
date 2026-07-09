"""
Custom data ingestion pipeline for CausalOCPM.

Production-grade "Upload Your Own Data" support: parsing, a 6-check data-quality
audit with a 0-100 score, smart imputation/cleaning, and confidence scoring.

The same five-phase pipeline (object graph → causal discovery → mixed SCM →
do-operator → attribution) runs on uploaded data unchanged — this module only
prepares and grades arbitrary event logs so they can enter that pipeline safely.

Design principle: surface data-quality problems explicitly rather than silently
producing unreliable causal estimates ("garbage in" must be visible, not hidden).
"""

from __future__ import annotations

import io
import json
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are identifiers / timestamps, never modelled as numeric features
_NON_FEATURE_COLS = {
    "case_id", "order_id", "patient_id", "timestamp", "time", "date",
    "activity", "event_id", "_cleaning_log", "case:concept:name",
    "concept:name", "time:timestamp",
}


# ── PARSING ───────────────────────────────────────────────────────────────────

def parse_uploaded_file(file_obj: Any) -> pd.DataFrame:
    """
    Parse an uploaded event log into a flat DataFrame.

    Supports CSV and OCEL 2.0 JSON. For OCEL JSON, event attributes are
    flattened into one row per event. Raises ValueError with a readable message
    on unrecognised or unparseable input.
    """
    name = (getattr(file_obj, "name", "") or "").lower()

    if name.endswith(".csv"):
        return _parse_csv(file_obj)
    if name.endswith(".json") or name.endswith(".jsonocel"):
        return _parse_ocel_json(file_obj)

    # Unknown extension — attempt CSV, then JSON
    try:
        return _parse_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        return _parse_ocel_json(file_obj)


def _parse_csv(file_obj: Any) -> pd.DataFrame:
    """Read a CSV upload, parsing a timestamp column if present."""
    file_obj.seek(0)
    df = pd.read_csv(file_obj)
    for ts_col in ("timestamp", "time:timestamp", "time", "date"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            break
    if df.shape[1] == 0:
        raise ValueError("CSV contains no columns.")
    return df


def _parse_ocel_json(file_obj: Any) -> pd.DataFrame:
    """Flatten an OCEL 2.0 / generic JSON event log into one row per event."""
    file_obj.seek(0)
    raw = file_obj.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    obj = json.loads(raw)

    # OCEL 2.0 standard: {"ocel:events": {...}} or {"events": [...]}
    events = None
    if isinstance(obj, dict):
        if "ocel:events" in obj:
            ev = obj["ocel:events"]
            events = list(ev.values()) if isinstance(ev, dict) else ev
        elif "events" in obj:
            events = obj["events"]
    elif isinstance(obj, list):
        events = obj

    if not events:
        raise ValueError("No events found in JSON (expected 'ocel:events' or 'events').")

    rows = []
    for ev in events:
        row: Dict[str, Any] = {}
        attrs = ev.get("ocel:vmap", ev.get("attributes", {})) if isinstance(ev, dict) else {}
        if isinstance(attrs, dict):
            row.update(attrs)
        if isinstance(ev, dict):
            if "ocel:timestamp" in ev:
                row["timestamp"] = ev["ocel:timestamp"]
            if "ocel:activity" in ev:
                row["activity"] = ev["ocel:activity"]
        rows.append(row)

    df = pd.json_normalize(rows)
    if df.empty:
        raise ValueError("OCEL JSON parsed to an empty table.")
    return df


# ── DETECTION HELPERS ─────────────────────────────────────────────────────────

def detect_binary_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns whose non-null values are a subset of {0, 1}."""
    binary = []
    for col in df.select_dtypes(include="number").columns:
        vals = set(pd.unique(df[col].dropna()))
        if vals and vals.issubset({0, 1, 0.0, 1.0}):
            binary.append(col)
    return binary


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric, model-eligible feature columns (identifiers excluded)."""
    return [
        c for c in df.columns
        if c.lower() not in _NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def compute_confidence(quality_score: int, n_rows: int) -> int:
    """
    Map quality score + sample size to a 0-95 result-confidence percentage.

    Larger samples add a bonus (more data → more reliable causal discovery),
    capped so that no uploaded dataset is ever presented as fully certain.
    """
    row_bonus = min(20, n_rows // 100)
    return int(min(95, max(0, quality_score) + row_bonus))


# ── QUALITY AUDIT ─────────────────────────────────────────────────────────────

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Six-check data-quality audit returning a 0-100 score and recommendations.

    Checks: (1) sample size, (2) missing values, (3) low variance,
    (4) duplicates, (5) numeric-column count, (6) outliers. Each problem
    deducts from a starting score of 100. ``issues`` are blocking;
    ``warnings`` are non-blocking and trigger automatic cleaning.
    """
    report: Dict[str, Any] = {}
    issues: List[str] = []
    warnings: List[str] = []
    score = 100

    n_rows, n_cols = df.shape
    report["n_rows"] = int(n_rows)
    report["n_cols"] = int(n_cols)

    # 1. SIZE
    if n_rows < 50:
        issues.append(f"Only {n_rows} rows — minimum 50 needed for the PC algorithm "
                      "to discover structure reliably.")
        score -= 40
    elif n_rows < 100:
        warnings.append(f"{n_rows} rows — results will be less reliable; 200+ recommended.")
        score -= 15
    elif n_rows < 200:
        warnings.append(f"{n_rows} rows — acceptable, but 500+ gives better causal discovery.")
        score -= 5

    # 2. MISSING VALUES
    missing_pct = (df.isnull().sum() / max(len(df), 1) * 100).round(1)
    cols_high_missing = missing_pct[missing_pct > 30].index.tolist()
    cols_some_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 30)].index.tolist()
    report["missing_summary"] = missing_pct.to_dict()
    report["cols_high_missing"] = cols_high_missing
    report["cols_some_missing"] = cols_some_missing
    if cols_high_missing:
        issues_or_warn = f"Columns >30% missing (will be dropped): {', '.join(cols_high_missing)}"
        warnings.append(issues_or_warn)
        score -= 20
    if cols_some_missing:
        warnings.append(f"Columns with some missing values (will be imputed): "
                        f"{', '.join(cols_some_missing)}")
        score -= 5

    # 3. LOW VARIANCE
    numeric_df = df.select_dtypes(include="number")
    low_variance_cols = []
    for col in numeric_df.columns:
        col_mean = numeric_df[col].mean()
        cv = numeric_df[col].std() / (abs(col_mean) + 1e-8)
        if pd.notna(cv) and cv < 0.01:
            low_variance_cols.append(col)
    report["low_variance_cols"] = low_variance_cols
    if low_variance_cols:
        warnings.append(f"Near-constant columns (will be excluded): {', '.join(low_variance_cols)}")
        score -= 10

    # 4. DUPLICATES
    n_duplicates = int(df.duplicated().sum())
    report["n_duplicates"] = n_duplicates
    if n_duplicates > len(df) * 0.1:
        warnings.append(f"{n_duplicates} duplicate rows "
                        f"({n_duplicates / max(len(df),1) * 100:.0f}%) — will be removed.")
        score -= 10

    # 5. NUMERIC COLUMN COUNT (exclude identifiers)
    n_numeric = len(feature_columns(df))
    report["n_numeric_features"] = n_numeric
    if n_numeric < 3:
        issues.append(f"Only {n_numeric} numeric feature columns — need at least 3 "
                      "for meaningful causal discovery.")
        score -= 30

    # 6. OUTLIERS
    outlier_cols = []
    for col in numeric_df.columns:
        if numeric_df[col].notna().sum() < 10:
            continue
        q1, q99 = numeric_df[col].quantile([0.01, 0.99])
        extreme = ((numeric_df[col] < q1) | (numeric_df[col] > q99)).sum()
        if extreme > len(df) * 0.05:
            outlier_cols.append(col)
    report["outlier_cols"] = outlier_cols
    if outlier_cols:
        warnings.append(f"Significant outliers (will be winsorized): {', '.join(outlier_cols)}")
        score -= 5

    report["score"] = max(0, score)
    report["issues"] = issues
    report["warnings"] = warnings
    report["can_proceed"] = len(issues) == 0
    return report


# ── CLEANING ──────────────────────────────────────────────────────────────────

def clean_custom_data(df: pd.DataFrame,
                       quality_report: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Auto-clean uploaded data per the quality report; returns (clean_df, log).

    Steps: drop duplicates → drop high-missing columns → drop low-variance
    columns → median-impute numeric / mode-impute categorical → winsorize
    outliers → ordinal-encode categoricals. Every action is recorded in the
    returned log for full transparency in the UI.
    """
    df = df.copy()
    log: List[str] = []

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df) > 0:
        log.append(f"Removed {before - len(df)} duplicate rows")

    # 2. High-missing columns
    high_missing = [c for c in quality_report.get("cols_high_missing", []) if c in df.columns]
    if high_missing:
        df = df.drop(columns=high_missing, errors="ignore")
        log.append(f"Dropped {len(high_missing)} high-missing columns: {', '.join(high_missing)}")

    # 3. Low-variance columns
    low_var = [c for c in quality_report.get("low_variance_cols", []) if c in df.columns]
    if low_var:
        df = df.drop(columns=low_var, errors="ignore")
        log.append(f"Dropped {len(low_var)} near-constant columns: {', '.join(low_var)}")

    # 4. Imputation
    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.append(f"Imputed {n_missing} missing in '{col}' with median ({median_val:.2f})")

    for col in cat_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            mode = df[col].mode()
            mode_val = mode.iloc[0] if not mode.empty else "UNKNOWN"
            df[col] = df[col].fillna(mode_val)
            log.append(f"Imputed {n_missing} missing in '{col}' with mode ('{mode_val}')")

    # 5. Winsorize outliers
    for col in quality_report.get("outlier_cols", []):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            q01, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q01, upper=q99)
            log.append(f"Winsorized outliers in '{col}' to [{q01:.2f}, {q99:.2f}]")

    # 6. Ordinal-encode remaining categoricals (skip identifier-like columns)
    for col in cat_cols:
        if col in df.columns and col.lower() not in _NON_FEATURE_COLS:
            df[col] = pd.Categorical(df[col]).codes.astype(float)
            log.append(f"Encoded categorical '{col}' as ordinal integers")

    return df, log


if __name__ == "__main__":
    # Smoke test on a small synthetic frame
    rng = np.random.default_rng(42)
    n = 300
    demo = pd.DataFrame({
        "complexity": rng.integers(1, 10, n).astype(float),
        "express": rng.binomial(1, 0.3, n).astype(float),
        "lead_time": rng.normal(5, 2, n),
        "delay": rng.normal(8, 3, n),
    })
    q = analyze_data_quality(demo)
    print(f"[custom_loader] score={q['score']} can_proceed={q['can_proceed']} "
          f"binary={detect_binary_columns(demo)} features={feature_columns(demo)}")
    _, lg = clean_custom_data(demo, q)
    print(f"[custom_loader] cleaning steps: {len(lg)}")
