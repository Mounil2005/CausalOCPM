# CausalOCPM — Dataset Summary (v2 — Realism Update)
**Project:** Causal-Explainable Object-Centric Process Mining  
**Author:** Placement Project  
**Date:** June 2026  
**Version:** v2 adds outliers, irregular timestamps, concept drift, seasonal effects

---

## Overview

Two synthetic datasets are provided, each with **15,000 rows** and a **planted causal structure**.  
The ground truth is known, which allows rigorous validation of the causal pipeline.

Both datasets follow the same confounding pattern:
> A complexity variable drives both treatment assignment AND outcome directly,  
> causing naive analysis to overstate the treatment's true causal effect.

---

## Dataset 1 — Manufacturing (`prihir_synthetic.csv`)

**Context:** Prihir Enterprises — brass parts manufacturer, Jamnagar  
**Rows:** 15,000 orders  
**Columns:** 14  
**Time Range:** 2023-01-01, one event every 6 hours

### Columns

| Column | Type | Role | Description |
|--------|------|------|-------------|
| `order_id` | ID | Case | Unique order identifier (ORD_0000 … ORD_14999) |
| `order_complexity` | Continuous (1–10) | **Confounder** | Complexity of the order; drives both supplier choice and delay |
| `supplier_a` | Binary (0/1) | **Treatment** | 1 = Supplier A used, 0 = Supplier B used |
| `material_lead_time` | Continuous (days) | Mediator | Days from order to material arrival |
| `machine_queue_length` | Continuous | Mediator | Number of orders ahead in machine queue |
| `export_flag` | Binary (0/1) | Independent | 1 = export order (extra approval needed) |
| `approval_duration` | Continuous (days) | Mediator | Days spent in approval process |
| `carrier_express` | Binary (0/1) | Independent | 1 = express carrier used |
| `shipment_delay` | Continuous (days) | **Outcome** | Total shipment delay in days |
| `machine_id` | ID (MCH_01–08) | Object | Which machine processed the order |
| `worker_id` | ID (WRK_01–15) | Object | Which worker handled the order |
| `material_id` | MAT_A / MAT_B | Object | Material type (linked to supplier) |
| `shipment_id` | ID (SHP_0000) | Object | Unique shipment identifier |
| `timestamp` | datetime | Timeline | Event timestamp |

### Key Statistics

| Metric | Value |
|--------|-------|
| Average Shipment Delay | **6.24 days** |
| Std Deviation | 4.44 days |
| Min Delay | −1.90 days |
| Max Delay | ~25+ days (outlier cases) |
| Supplier A orders | 8,164 (54.4%) |
| Supplier B orders | 6,836 (45.6%) |
| Outlier rows (±3σ) | ~111 rows (0.7%) |
| Supplier failure rows | 150 rows (MLT > 15 days) |
| Regulatory hold rows | 150 rows (Approval > 25 days) |
| Q4 avg delay | 6.91 days (+0.9 vs non-Q4) |
| Timestamp gap range | 0.3h – 71.7h (irregular) |
| Timestamp avg gap | 9.74h (business-hour weighted) |
| Concept drift | Noise std increases after row 10,000 |

### Planted Causal Structure (Ground Truth — 9 edges)

```
order_complexity  ──►  supplier_a               (confounder drives treatment)
order_complexity  ──►  machine_queue_length
order_complexity  ──►  shipment_delay            (backdoor / confounding path)
supplier_a        ──►  material_lead_time         β = +7.4 days
material_lead_time ──►  shipment_delay            β = +0.9
machine_queue_length ──►  approval_duration       β = +1.3
export_flag       ──►  approval_duration          β = +2.0
approval_duration ──►  shipment_delay             β = +0.1
carrier_express   ──►  shipment_delay             β = −0.6
```

### True vs Naive Effect

| Method | Estimated Effect |
|--------|-----------------|
| Naive (confounded) | ~+7.94 days |
| True causal effect (planted) | **+6.66 days** |
| Confounding bias | 1.28 days (19.3% overestimate) |

---

## Dataset 2 — Healthcare (`hospital_synthetic.csv`)

**Context:** Hospital admissions — length of stay analysis  
**Rows:** 15,000 patient admissions  
**Columns:** 14  
**Time Range:** 2023-01-01, one event every 4 hours

### Columns

| Column | Type | Role | Description |
|--------|------|------|-------------|
| `patient_id` | ID | Case | Unique patient identifier (PAT_0000 … PAT_14999) |
| `patient_complexity` | Continuous (1–10) | **Confounder** | Clinical complexity; drives specialist assignment and LOS |
| `specialist_required` | Binary (0/1) | **Treatment** | 1 = specialist assigned, 0 = general practitioner |
| `treatment_duration` | Continuous (days) | Mediator | Duration of active treatment |
| `bed_occupancy_rate` | Continuous (0–1) | Mediator | Ward occupancy at time of admission |
| `emergency_admission` | Binary (0/1) | Independent | 1 = emergency admission |
| `approval_wait` | Continuous (days) | Mediator | Days waiting for treatment approval |
| `insurance_expedited` | Binary (0/1) | Independent | 1 = insurance expedited processing |
| `length_of_stay` | Continuous (days) | **Outcome** | Total hospital length of stay |
| `ward_id` | ID (WRD_01–06) | Object | Which ward the patient was admitted to |
| `clinician_id` | ID (CLN_01–20) | Object | Which clinician managed the case |
| `medication_id` | MED_S / MED_G | Object | Medication type (specialist vs general) |
| `discharge_id` | ID (DIS_0000) | Object | Unique discharge record identifier |
| `timestamp` | datetime | Timeline | Admission timestamp |

### Key Statistics

| Metric | Value |
|--------|-------|
| Average Length of Stay | **4.99 days** |
| Std Deviation | 3.51 days |
| Min LOS | −1.65 days |
| Max LOS | ~18+ days (outlier cases) |
| Specialist cases | 8,164 (54.4%) |
| Non-specialist cases | 6,836 (45.6%) |
| Outlier rows (±3σ) | ~120 rows (0.8%) |
| ICU escalation rows | 150 rows (Treatment > 12 days) |
| Insurance dispute rows | 150 rows (Approval wait > 18 days) |
| Winter avg LOS | 5.38 days (+0.6 vs other months) |
| Timestamp gap range | 0.2h – 47.9h (irregular) |
| Timestamp avg gap | 5.53h (ED-rush weighted) |
| Concept drift | Noise std increases after row 10,000 |

### Planted Causal Structure (Ground Truth — 9 edges)

```
patient_complexity   ──►  specialist_required        (confounder drives treatment)
patient_complexity   ──►  bed_occupancy_rate
patient_complexity   ──►  length_of_stay             (backdoor / confounding path)
specialist_required  ──►  treatment_duration          β = +6.2 days
treatment_duration   ──►  length_of_stay             β = +0.85
bed_occupancy_rate   ──►  approval_wait              β = +2.1
emergency_admission  ──►  approval_wait              β = +1.8
approval_wait        ──►  length_of_stay             β = +0.35
insurance_expedited  ──►  length_of_stay             β = −0.55
```

### True vs Naive Effect

| Method | Estimated Effect |
|--------|-----------------|
| Naive (confounded) | ~+6.09 days |
| True causal effect (planted) | **+5.27 days** |
| Confounding bias | 0.82 days (15.5% overestimate) |

---

## Sample Upload Files (`sample_uploads/`)

These smaller files are used to test the custom CSV upload feature:

| File | Rows | Purpose |
|------|------|---------|
| `test_good.csv` | 500 | Valid upload — 4 columns: complexity, express_carrier, lead_time, delay |
| `test_missing.csv` | — | Contains NaN values — tests missing data handling |
| `test_small.csv` | — | Too few rows — tests minimum size validation |
| `test_lowvar.csv` | — | Near-zero variance — tests variance check |

---

## How the Datasets Were Generated

Both datasets are **synthetically generated** using a known structural causal model (SCM).  
This means the ground truth causal graph and coefficients are known in advance,  
allowing the pipeline's discovery accuracy to be measured precisely.

**Generation code:**
- Manufacturing: `causal_ocpm/data/generate_data.py`
- Healthcare: `causal_ocpm/data/generate_healthcare.py`

To regenerate:
```bash
python causal_ocpm/data/generate_data.py
python causal_ocpm/data/generate_healthcare.py
```

---

## Realism Features (v2)

Both datasets now include four real-world data quality characteristics:

### 1. Outliers (~2% of rows)

| Domain | Outlier Type | Column Affected | Range |
|--------|-------------|-----------------|-------|
| Manufacturing | Supplier failure | `material_lead_time` | 18–30 days |
| Manufacturing | Regulatory hold | `approval_duration` | 30–60 days |
| Healthcare | ICU escalation | `treatment_duration` | 15–30 days |
| Healthcare | Insurance dispute | `approval_wait` | 20–45 days |

Outlier effects propagate into the outcome column through the structural equations, keeping the SCM internally consistent.

### 2. Irregular Timestamps

**Manufacturing** — Business-hour weighted event gaps:
- Normal working hours (Mon–Fri 08–18): avg ~3.6h gaps
- Nights/weekends: avg ~12.5h gaps
- Burst (urgent orders): 0.3h – 1.5h
- Shutdown/holiday: 26h – 72h

**Healthcare** — ED-rush weighted admission gaps:
- Weekday daytime (07–22): avg ~3.3h gaps
- Nights: avg ~8h gaps
- Weekends: avg ~7.5h gaps
- ED surge: 0.2h – 1.0h
- Holiday idle: 24h – 48h

### 3. Concept Drift

After row 10,000 (approx. mid-2024 in the timeline), outcome noise standard deviation increases by ~30%:
- **Manufacturing**: factory expansion with teething problems
- **Healthcare**: new ward opened with junior staff

The structural causal coefficients remain unchanged — only the residual noise grows, simulating increased process variability without changing root causes.

### 4. Seasonal Effects

| Domain | Season | Effect |
|--------|--------|--------|
| Manufacturing | Q4 (Oct–Dec) | +0.8 days avg shipment delay (peak demand) |
| Healthcare | Winter (Nov–Feb) | +0.6 days avg length of stay (flu season) |

### 5. New Categorical Column

| Domain | Column | Values |
|--------|--------|--------|
| Manufacturing | `event_type` | order_placed, material_received, production_started, quality_check, shipment_dispatched |
| Healthcare | `admission_type` | elective, emergency, transfer, day_case, maternity |

---

## Pipeline Validation Results

| Metric | Manufacturing | Healthcare |
|--------|--------------|------------|
| DAG Discovery F1 (full 15K) | **1.000** | **1.000** |
| DAG Discovery F1 (stress n=500) | 0.80 | 0.67 |
| Causal Effect Recovery Error | 0.38 days | 0.30 days |
| Bootstrap Stability | 85% | 68% |
| Estimation Method | Double ML (cross-fitted GBM) | Double ML (cross-fitted GBM) |
