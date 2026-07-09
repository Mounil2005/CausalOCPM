"""
BPI Challenge 2019 real data converter for CausalOCPM.

Converts the BPI Challenge 2019 XES event log (purchase orders from a major
Dutch manufacturer) into the CausalOCPM OCEL schema. Used in Tab 6 for
real-world validation — eliminating the circular validation objection.

NOTE: Phases 1-2 only are applied to real data. Phase 4 (causal effect
estimation) is NOT run on BPI 2019 because no ground truth causal structure
is available. This is scientifically correct and is acknowledged explicitly
in the dashboard rather than obscured.

Dataset:
  BPI Challenge 2019 — Purchase order handling process
  URL: https://doi.org/10.4121/uuid:d06aff4b-79f0-45ab-b737-5954ad1dac79
  Format: XES (IEEE standard event log format)
  Approx size: 250MB–500MB

To enable this tab:
  1. Download BPI_Challenge_2019.xes from the URL above
  2. Place in the data/ directory
  3. Run: python data/convert_bpi2019.py
"""

import warnings
import logging
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
XES_PATH = DATA_DIR / "BPI_Challenge_2019.xes"
DATA_PATH = DATA_DIR / "bpi2019_converted.csv"
PROC_VARS_PATH = DATA_DIR / "bpi2019_proc_vars.csv"

DATASET_URL = "https://doi.org/10.4121/uuid:d06aff4b-79f0-45ab-b737-5954ad1dac79"


def map_bpi2019_to_ocel(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map BPI 2019 columns to CausalOCPM's 4-role OCEL schema.

    BPI 2019 records purchase order events. We map each column to the
    equivalent OCEL object role used throughout the framework.
    """
    df = pd.DataFrame()

    # Case / Order (primary object)
    case_col = _find_column(df_raw, ['case:concept:name', 'case_id', 'CaseID'])
    df['order_id'] = df_raw[case_col].astype(str) if case_col else _synthetic_ids('ORD', len(df_raw))

    # Activity
    act_col = _find_column(df_raw, ['concept:name', 'Activity', 'activity'])
    df['activity'] = df_raw[act_col].astype(str) if act_col else 'UNKNOWN'

    # Timestamp
    ts_col = _find_column(df_raw, ['time:timestamp', 'timestamp', 'Timestamp'])
    if ts_col:
        df['timestamp'] = pd.to_datetime(df_raw[ts_col], utc=True, errors='coerce')
    else:
        df['timestamp'] = pd.NaT

    # Resource / Worker
    res_col = _find_column(df_raw, ['org:resource', 'Resource', 'resource'])
    if res_col:
        df['worker_id'] = df_raw[res_col].astype(str)
    else:
        df['worker_id'] = _synthetic_ids('WRK', len(df_raw))

    # Vendor (Artifact role)
    vendor_col = _find_column(df_raw, ['case:vendor', 'Vendor', 'vendor'])
    if vendor_col:
        df['vendor_id'] = df_raw[vendor_col].astype(str)
    else:
        df['vendor_id'] = 'VND_UNKNOWN'

    # Document type
    doc_col = _find_column(df_raw, ['case:document type', 'case:Document Type',
                                     'document_type'])
    df['doc_type'] = df_raw[doc_col].astype(str) if doc_col else 'UNKNOWN'

    # Goods receipt (binary)
    gr_col = _find_column(df_raw, ['case:Goods Receipt', 'case:goods_receipt'])
    if gr_col:
        df['has_goods_receipt'] = df_raw[gr_col].map(
            {'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
    else:
        df['has_goods_receipt'] = 0

    # Item type
    item_col = _find_column(df_raw, ['case:Item Type', 'case:item_type'])
    df['item_type'] = df_raw[item_col].astype(str) if item_col else 'UNKNOWN'

    # Spending authority → ordinal 1–5
    spend_col = _find_column(df_raw, ['case:spending authority level',
                                       'case:Spending Authority Level',
                                       'spending_authority'])
    if spend_col:
        spend_vals = pd.to_numeric(df_raw[spend_col], errors='coerce').fillna(3)
        df['spending_level'] = spend_vals.clip(1, 5).astype(int)
    else:
        df['spending_level'] = 3

    # Synthesise OCEL role columns
    df['machine_id'] = df['worker_id'].apply(
        lambda r: f"MCH_{int(hashlib.md5(str(r).encode()).hexdigest(), 16) % 12 + 1:02d}"
    )
    df['shipment_id'] = 'SHP_' + df['order_id'].astype(str)
    df['material_id'] = 'MAT_' + df['vendor_id'].str[:3]

    # Drop rows with null timestamps or null case IDs
    df = df.dropna(subset=['timestamp', 'order_id'])
    df = df[df['order_id'] != 'nan']

    return df.reset_index(drop=True)


def get_bpi2019_process_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive case-level process variables from BPI 2019 for Phase 2 discovery.

    Aggregates per case_id to produce one row per purchase order with
    variables that mirror the manufacturing domain's causal structure.
    This allows Phase 2 (causal discovery) to run identically on both
    synthetic and real data.
    """
    case_groups = df.groupby('order_id')

    proc_vars = pd.DataFrame()
    proc_vars['order_id'] = list(case_groups.groups.keys())

    # Number of events per case ≈ order complexity
    proc_vars['order_complexity'] = case_groups.size().values.clip(1, 20)

    # High spending level → specialist / high-authority supplier
    proc_vars['has_specialist'] = (
        case_groups['spending_level'].max().values >= 4).astype(int)

    # Processing time in days (first to last event)
    time_range = case_groups['timestamp'].agg(lambda x: (x.max() - x.min()).days)
    proc_vars['processing_time'] = time_range.values.clip(0, 365)

    # Number of distinct activities per case
    act_count = case_groups['activity'].nunique()
    proc_vars['activity_count'] = act_count.values

    # Vendor diversity
    vendor_count = case_groups['vendor_id'].nunique()
    proc_vars['vendor_count'] = vendor_count.values.clip(1, 10)

    # Goods receipt presence
    proc_vars['has_goods_receipt'] = case_groups['has_goods_receipt'].max().values

    # Cycle time = total case duration in days (outcome variable)
    cycle_time = case_groups['timestamp'].agg(lambda x: (x.max() - x.min()).days)
    proc_vars['cycle_time'] = cycle_time.values.clip(0, 365)

    return proc_vars.reset_index(drop=True)


def try_load_bpi2019() -> pd.DataFrame | None:
    """
    Attempt to load BPI 2019 XES file using pm4py. Returns None if not available.

    Graceful degradation: the dashboard Tab 6 renders download instructions
    when this returns None rather than raising an error.
    """
    # Try loading from pre-converted CSV first (much faster)
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
            logger.info(f"[BPI 2019] Loaded from converted CSV. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"[BPI 2019] CSV load failed: {e}")

    # Try loading from XES
    if not XES_PATH.exists():
        logger.warning(f"[BPI 2019] XES file not found at {XES_PATH}")
        return None

    try:
        import pm4py
        logger.info("[BPI 2019] Loading XES file (this may take 2-3 minutes)...")
        log = pm4py.read_xes(str(XES_PATH))
        df_raw = pm4py.convert_to_dataframe(log)
        logger.info(f"[BPI 2019] Raw log loaded. Shape: {df_raw.shape}")

        df = map_bpi2019_to_ocel(df_raw)
        df.to_csv(DATA_PATH, index=False)
        logger.info(f"[BPI 2019] Converted and saved. Shape: {df.shape}")
        return df

    except ImportError:
        logger.warning("[BPI 2019] pm4py not installed — cannot load XES file.")
        return None
    except Exception as e:
        logger.warning(f"[BPI 2019] Load failed: {e}")
        return None


def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    """Return the first matching column name from candidates list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _synthetic_ids(prefix: str, n: int) -> list:
    return [f"{prefix}_{i:04d}" for i in range(n)]


if __name__ == "__main__":
    df = try_load_bpi2019()
    if df is not None:
        print(f"[BPI 2019] Loaded and converted. Shape: {df.shape}")
        proc_vars = get_bpi2019_process_variables(df)
        print(f"[BPI 2019] Process variables derived. Shape: {proc_vars.shape}")
    else:
        print("[BPI 2019] File not found. Download from:")
        print(f"  {DATASET_URL}")
        print("  Place BPI_Challenge_2019.xes in the data/ directory and rerun.")
