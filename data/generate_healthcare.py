"""
Healthcare domain data generator for CausalOCPM.

Second domain instantiation proving the "generalised framework" claim.
Mirrors the manufacturing confounding structure in a hospital admissions setting:

The confounding trap (healthcare):
  patient_complexity → specialist_required (complex patients need specialists)
  patient_complexity → length_of_stay (complex patients stay longer inherently)
  specialist_required → treatment_duration → length_of_stay (true causal path)

  Naive analysis overstates specialist's contribution to length of stay.
  Causal adjustment recovers the true planted coefficient.

Realism features added (v2):
  - ~2% outlier rows simulating ICU escalations and insurance disputes
  - Irregular admission timestamps (ED rushes, quiet nights, weekends)
  - Mild concept drift: ward noise increases after row 10,000 (new wing opened)
  - Winter seasonal pressure: Nov–Feb admissions carry +0.6 day average LOS
"""

import numpy as np
import pandas as pd
from pathlib import Path


DOMAIN = "healthcare"

GROUND_TRUTH_EDGES = [
    ("patient_complexity", "specialist_required"),
    ("patient_complexity", "bed_occupancy_rate"),
    ("patient_complexity", "length_of_stay"),
    ("specialist_required", "treatment_duration"),
    ("treatment_duration", "length_of_stay"),
    ("bed_occupancy_rate", "approval_wait"),
    ("approval_wait", "length_of_stay"),
    ("emergency_admission", "approval_wait"),
    ("insurance_expedited", "length_of_stay"),
]

# True planted causal coefficient — used ONLY in test suite, never in UI
TRUE_SPECIALIST_CAUSAL_EFFECT = 6.2 * 0.85

NUMERIC_VARS = [
    'patient_complexity', 'specialist_required', 'treatment_duration',
    'bed_occupancy_rate', 'emergency_admission', 'approval_wait',
    'insurance_expedited', 'length_of_stay'
]

BINARY_VARS = ['specialist_required', 'emergency_admission', 'insurance_expedited']
CONTINUOUS_VARS = ['patient_complexity', 'treatment_duration',
                   'bed_occupancy_rate', 'approval_wait']
OUTCOME_VAR = 'length_of_stay'
TREATMENT_VAR = 'specialist_required'

DATA_PATH = Path(__file__).parent / "hospital_synthetic.csv"

VARIABLE_LABELS = {
    'patient_complexity':   'Patient Complexity',
    'specialist_required':  'Specialist Required',
    'treatment_duration':   'Treatment Duration (days)',
    'bed_occupancy_rate':   'Bed Occupancy Rate',
    'emergency_admission':  'Emergency Admission',
    'approval_wait':        'Approval Wait (days)',
    'insurance_expedited':  'Insurance Expedited',
    'length_of_stay':       'Length of Stay (days)',
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _generate_admission_timestamps(n: int, start: str = "2023-01-01",
                                   rng: np.random.Generator = None) -> pd.DatetimeIndex:
    """
    Generate realistic hospital admission timestamps.

    Admission gap distribution:
      - Weekday daytime (07:00–22:00): short gaps ~2–5h (high ED throughput)
      - Nights (22:00–07:00): longer gaps ~6–14h (quiet period)
      - Weekends: slightly longer gaps (reduced elective admissions)
      - ~5% ED rush: burst gap < 1h (mass casualty / flu surge)
      - ~2% long gap: > 24h (public holiday, reduced staffing)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    timestamps = [pd.Timestamp(start)]
    for _ in range(n - 1):
        last    = timestamps[-1]
        weekday = last.weekday()
        hour    = last.hour

        is_daytime  = 7 <= hour < 22
        is_weekend  = weekday >= 5

        roll = rng.random()
        if roll < 0.05:
            # ED rush / flu surge burst
            gap_h = rng.uniform(0.2, 1.0)
        elif roll < 0.07:
            # Holiday / reduced staffing idle
            gap_h = rng.uniform(24, 48)
        elif is_daytime and not is_weekend:
            # Normal weekday admissions
            gap_h = rng.gamma(shape=2.2, scale=1.5)   # mean ~3.3h
            gap_h = np.clip(gap_h, 0.5, 10.0)
        elif is_weekend:
            # Weekend: fewer elective admissions
            gap_h = rng.gamma(shape=2.5, scale=3.0)   # mean ~7.5h
            gap_h = np.clip(gap_h, 2.0, 20.0)
        else:
            # Night: quiet period
            gap_h = rng.gamma(shape=2.0, scale=4.0)   # mean ~8h
            gap_h = np.clip(gap_h, 3.0, 18.0)

        timestamps.append(last + pd.Timedelta(hours=float(gap_h)))

    return pd.DatetimeIndex(timestamps)


def generate_data(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic hospital admission event log with planted causal structure.

    Analogous to manufacturing domain: patient_complexity is the confounder
    driving both specialist assignment and length of stay directly.

    Realism additions (do not alter the causal structure):
      1. Outliers  — 2% of rows simulate ICU escalations (long treatment_duration)
                     and insurance disputes (long approval_wait).
      2. Irregular timestamps — ED-rush weighted gaps instead of uniform 4h.
      3. Concept drift — outcome noise std increases by 30% after row 10,000
                         simulating the opening of a new ward with junior staff.
      4. Winter pressure — Nov–Feb admissions carry +0.6 day average LOS.
    """
    rng = np.random.default_rng(seed)

    # ── Core causal generation ────────────────────────────────────────────────

    patient_complexity = rng.integers(1, 11, size=n).astype(float)

    specialist_prob = _sigmoid((patient_complexity - 5) * 0.9)
    specialist_required = rng.binomial(1, specialist_prob).astype(float)

    treatment_duration = (1.8 + 6.2 * specialist_required
                          + rng.normal(0, 0.7, size=n))
    treatment_duration = np.clip(treatment_duration, 0.3, None)

    bed_occupancy_rate = (0.3 + 0.06 * patient_complexity
                          + rng.normal(0, 0.08, size=n))
    bed_occupancy_rate = np.clip(bed_occupancy_rate, 0, 1)

    emergency_admission = rng.binomial(1, 0.4, size=n).astype(float)

    approval_wait = (3.2 + 2.1 * bed_occupancy_rate
                     + 1.8 * emergency_admission
                     + rng.normal(0, 0.9, size=n))
    approval_wait = np.clip(approval_wait, 0, None)

    insurance_expedited = rng.binomial(1, 0.45, size=n).astype(float)

    # ── Concept drift: ward noise grows after row 10,000 ─────────────────────
    base_noise  = rng.normal(0, 0.5, size=n)
    drift_noise = rng.normal(0, 0.65, size=n)
    drift_mask  = np.arange(n) >= 10_000
    outcome_noise = np.where(drift_mask, drift_noise, base_noise)

    # ── Outcome: length of stay ───────────────────────────────────────────────
    length_of_stay = (0.8
                      + 0.85 * (treatment_duration - 1.8)
                      + 0.35 * (approval_wait - 3.2) * 0.3
                      + 0.18 * patient_complexity
                      - 0.55 * insurance_expedited
                      + outcome_noise)

    # ── Realistic timestamps ──────────────────────────────────────────────────
    timestamps = _generate_admission_timestamps(n, start="2023-01-01", rng=rng)

    # ── Winter seasonal pressure: Nov–Feb +0.6 day LOS ───────────────────────
    months = pd.DatetimeIndex(timestamps).month
    winter_mask = np.isin(months, [11, 12, 1, 2])
    seasonal_bump = np.where(winter_mask, rng.normal(0.6, 0.25, size=n), 0.0)
    length_of_stay = length_of_stay + seasonal_bump

    # ── Outliers: ~2% of rows ─────────────────────────────────────────────────
    n_outliers = max(1, int(n * 0.02))
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)

    # ICU escalation: treatment_duration spikes to 15–30 days
    icu_idx = outlier_idx[:n_outliers // 2]
    treatment_duration[icu_idx] = rng.uniform(15, 30, size=len(icu_idx))

    # Insurance dispute: approval_wait spikes to 20–45 days
    dispute_idx = outlier_idx[n_outliers // 2:]
    approval_wait[dispute_idx] = rng.uniform(20, 45, size=len(dispute_idx))

    # Propagate outlier effects into LOS (keeps SCM consistent)
    length_of_stay[icu_idx] += (
        0.85 * (treatment_duration[icu_idx] - 1.8)
        - 0.85 * (1.8 + 6.2 * specialist_required[icu_idx] - 1.8)
    )
    length_of_stay[dispute_idx] += (
        0.35 * (approval_wait[dispute_idx] - 3.2) * 0.3
        - 0.35 * (3.2 + 2.1 * bed_occupancy_rate[dispute_idx]
                  + 1.8 * emergency_admission[dispute_idx] - 3.2) * 0.3
    )

    # ── Object-centric columns ────────────────────────────────────────────────
    patient_ids   = [f"PAT_{i:04d}" for i in range(n)]
    ward_ids      = [f"WRD_{(i % 6) + 1:02d}" for i in range(n)]
    clinician_ids = [f"CLN_{(i % 20) + 1:02d}" for i in range(n)]
    medication_ids = ["MED_S" if s == 1 else "MED_G" for s in specialist_required]
    discharge_ids = [f"DIS_{i:04d}" for i in range(n)]

    # ── Admission type column (adds realism for process mining) ───────────────
    admission_types = rng.choice(
        ["elective", "emergency", "transfer", "day_case", "maternity"],
        size=n, p=[0.35, 0.30, 0.10, 0.15, 0.10]
    )

    df = pd.DataFrame({
        'patient_id':          patient_ids,
        'patient_complexity':  patient_complexity,
        'specialist_required': specialist_required,
        'treatment_duration':  treatment_duration,
        'bed_occupancy_rate':  bed_occupancy_rate,
        'emergency_admission': emergency_admission,
        'approval_wait':       approval_wait,
        'insurance_expedited': insurance_expedited,
        'length_of_stay':      length_of_stay,
        'ward_id':             ward_ids,
        'clinician_id':        clinician_ids,
        'medication_id':       medication_ids,
        'discharge_id':        discharge_ids,
        'timestamp':           timestamps,
        'admission_type':      admission_types,
    })

    return df


def load_or_generate(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Load cached CSV if present, otherwise generate and save."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = generate_data(n=n, seed=seed)
    df.to_csv(DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    df = generate_data()
    df.to_csv(DATA_PATH, index=False)
    naive = (df[df.specialist_required == 1].length_of_stay.mean()
             - df[df.specialist_required == 0].length_of_stay.mean())
    print("[CONFOUNDING ANALYSIS — HEALTHCARE]")
    print(f"  Naive (confounded)     : +{naive:.3f} days LOS")
    print(f"  True causal effect     : +{TRUE_SPECIALIST_CAUSAL_EFFECT:.3f} days")
    print(f"  Confounding inflation  : {naive / TRUE_SPECIALIST_CAUSAL_EFFECT:.2f}x")
    print(f"  Data saved. Shape: {df.shape}")
