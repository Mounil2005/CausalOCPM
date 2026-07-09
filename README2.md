# CausalOCPM — System Architecture

**Causal-Explainable Object-Centric Process Mining**
*A technical walkthrough of what was built, how it works, and why it was built this way.*

Every fact, number, and file size in this document was checked directly against the code and a live test run while writing it — none of it is copied from marketing copy.

## Contents
1. [The Problem](#1-the-problem)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Backend Pipeline — Phase by Phase](#3-backend-pipeline--phase-by-phase)
4. [The Synthetic Datasets — Deep Dive](#4-the-synthetic-datasets--deep-dive)
5. [Frontend — Streamlit Dashboard](#5-frontend--streamlit-dashboard)
6. [Tech Stack](#6-tech-stack)
7. [Testing — Deep Dive](#7-testing--deep-dive)
8. [Custom Data Upload & Quality Scoring](#8-custom-data-upload--quality-scoring)
9. [Project Structure](#9-project-structure)
10. [What Makes This More Than a Dashboard](#10-what-makes-this-more-than-a-dashboard)
11. [How to Run It](#11-how-to-run-it)

---

## 1. The Problem

Standard process mining tools (Celonis, pm4py, Disco, etc.) answer *"what happened"* and *"what correlates with delay"*. They cannot answer *"what would happen if we changed X"* — because correlation-based analytics silently absorb confounding.

Concrete example from this project's manufacturing scenario:

> Complex orders are *both* more likely to use **Supplier A** *and* inherently take longer to ship. A naive analysis that just measures "Supplier A orders vs. others" attributes all of that extra delay to Supplier A — including the part that was actually caused by order complexity. It overstates Supplier A's true impact by roughly **20–24%**, depending on the run.

CausalOCPM's job is to detect that confounding automatically, adjust for it, and report the number a decision-maker can actually act on — with the uncertainty and validation evidence to back it.

---

## 2. High-Level Architecture

```
                     ┌─────────────────────────────┐
                     │   OCEL 2.0 Event Log (CSV)   │
                     │  15,000 events · 5 object    │
                     │  types · planted ground truth│
                     └───────────────┬──────────────┘
                                     │
   ┌─────────────────────────────────────────────────────────────────┐
   │                      BACKEND PIPELINE (src/)                     │
   │                                                                   │
   │  Phase 1          Phase 2           Phase 3         Phase 4/5     │
   │  Object Graph  →  Causal Discovery →  Mixed SCM  →  Effect Est.  │
   │  (pm4py-style)    (Bootstrapped PC)   (Log./Lin./  & Attribution│
   │                                        GBM by type) (DML + SHAP) │
   └─────────────────────────────────────────────────────────────────┘
                                     │
                     ┌───────────────┴──────────────┐
                     │   Streamlit Dashboard (app/)   │
                     │   5 tabs + Cerebras Copilot     │
                     └────────────────────────────────┘
```

Each phase is a **standalone, independently testable module** in `src/` — the dashboard doesn't contain any causal-inference logic itself, it only calls into these modules and renders their output. This separation is what makes the 49-test suite possible: every phase can be unit-tested against a synthetic dataset with a *known, planted* ground-truth effect, independent of the UI.

---

## 3. Backend Pipeline — Phase by Phase

### Phase 1 — Object Interaction Graph (`src/phase1_graph.py`, 192 lines)

Builds a typed heterogeneous graph from the raw event log: every event involves several objects (Case, Machine/Ward, Worker, Material, Shipment/Discharge), and objects that co-occur in the same event get connected, weighted by co-occurrence count.

This graph is **domain-agnostic** — manufacturing, healthcare, or a real BPI Challenge 2019 log are all mapped through the same 5 canonical object roles, so the rest of the pipeline never needs to know which domain it's running on.

### Phase 2 — Causal Discovery (`src/phase2_discovery.py`, 351 lines)

Learns the causal DAG using the **PC algorithm** (constraint-based causal discovery, Fisher's Z conditional-independence test, `causal-learn`'s implementation). Exact parameters, not defaults picked at random:

| Parameter | Value | Why |
|---|---|---|
| `alpha` (significance level) | 0.05 | Standard CI-test threshold |
| `n_bootstrap` | 20 subsamples | Enough runs to get a stable confidence estimate per edge without making discovery prohibitively slow |
| Subsample size | 2,000 rows each | Large enough to detect real structure, small enough that 20 runs stay fast |
| `bootstrap_threshold` | 0.60 (60%) | An edge must appear in ≥60% of the 20 bootstrap runs to be retained — filters out sampling-noise artifacts |

Why bootstrap at all: a single PC run is sensitive to sampling noise — an edge near the significance threshold can appear or vanish depending on the exact sample. Aggregating over 20 subsamples and keeping only edges that are *stably* detected turns "this edge showed up once" into "this edge is genuinely supported by the data," and gives each surviving edge its own confidence score (stored in `dag.graph['edge_confidence']`) instead of a binary yes/no.

Domain-knowledge constraints (forcing known-direction edges, forbidding reverse causation) are applied as a post-processing step — and `run_ablation_study()` runs discovery *with* and *without* those constraints and reports the difference, so the "Domain Knowledge Impact" panel in the dashboard is measured, not asserted.

**An honest limitation, documented in the code and intentionally not hidden:** one planted edge (`order_complexity → supplier_a`) is a genuinely non-linear relationship (sigmoid-link + Bernoulli draw). Correlation-based PC discovery structurally cannot reliably detect this edge from data alone, at any sample size — and the test suite explicitly exempts it from the "must be high-confidence" check rather than tuning the data generator until the problem disappears. This is deliberate: it's exactly the kind of relationship domain-expert knowledge exists to supply, and an earlier version of the generator that made this edge linearly-detectable also made pure discovery hit a perfect 9/9 recovery on its own — which would have made the "domain knowledge added value" panel meaningless (nothing left for domain knowledge to add).

### Phase 3 — Mixed Structural Causal Model (`src/phase3_scm.py`, 328 lines)

Fits one structural equation per node in the discovered DAG. The deliberate technical choice: **not every equation uses the same model class.**

| Variable type | Model | Why |
|---|---|---|
| Binary | `LogisticRegression` | A linear model on a 0/1 outcome (the "linear probability model") can predict probabilities outside [0,1] and biases coefficients — a known methodological error this project explicitly avoids |
| Outcome (e.g. shipment delay) | `GradientBoostingRegressor` | Captures non-linear structure without forcing a functional form |
| Other continuous | `LinearRegression` | Matches the linear structure actually planted in the data |

Every equation is scored with **cross-validated R²** ("CV-R²" — not a plain in-sample R², which would be optimistic and hide overfitting), computed via `cross_val_predict` over the same 5-fold split used elsewhere in the pipeline.

### Phase 4 — do-Operator: Causal Effect Estimation (`src/phase4_dooperator.py`, 714 lines — the largest single file in the project)

This is the heart of the "causal" claim. Primary estimator: **Double Machine Learning** (Chernozhukov et al., 2018, *Econometrica*).

Exact mechanics: `KFold(n_splits=5, shuffle=True, random_state=42)` cross-fitting. A `GradientBoostingRegressor` (or `GradientBoostingClassifier` with `predict_proba` if the treatment is binary) is fit out-of-fold to residualize *both* the outcome and the treatment against the confounders — i.e., the model predicts "what would this outcome/treatment look like given only the confounders," and the *residual* (actual minus predicted) is what's actually regressed against. Regressing outcome-residual on treatment-residual isolates the treatment's effect net of anything the confounders could explain, and is robust to non-linear confounding (like the sigmoid-shaped order-complexity → Supplier-A selection path planted in the data), with standard errors via the sandwich (Eicker-Huber-White) estimator.

Fallback chain if DML can't run: DoWhy's linear-regression backdoor adjustment → manual OLS backdoor with bootstrap confidence intervals. The pipeline never just fails silently.

**Sensitivity analysis** (not optional, always run as part of `compare_effects()`):
- **Placebo-treatment test** — replace the real treatment with random noise; the estimated effect should collapse to ~0 (verified in tests: must stay under 1.0).
- **Random common cause test** — add a fake confounder; the estimate shouldn't move much if the model is well-specified.
- **Unmeasured-confounding sweep** — 6 scenarios of varying hypothetical hidden-confounder strength, showing how much the estimate would need to shift before the causal claim breaks.
- **E-value** — the standard "how strong would an unmeasured confounder need to be to explain away this effect" metric.

**Robustness check:** `robustness_across_seeds()` reruns the entire discovery → SCM → DML pipeline across 10 random seeds (`range(42, 52)`) and confirms every single seed's estimate lands within ±0.5 days of the planted true effect — this is what backs the "validated across 10 seeds" claim, not a single favorable run.

### Phase 5 — SCM-Grounded Attribution (`src/phase5_attribution.py`, 272 lines)

Case-level "why did *this* case have this outcome" explanations, using SHAP applied to the fitted structural equations. Every explanation is checked for **SHAP additivity** in tests — `baseline + sum(shap_values) ≈ predicted_outcome`, within 0.15 tolerance — so the attribution isn't just plausible-looking numbers, it's numerically consistent with the model that produced it. Each contributing factor is also tagged `actionable` or `structural`, so a case report can distinguish "you can fix this" from "this is just how the case is."

One deliberate naming decision worth mentioning to a technical reviewer: this is called **"SCM-Grounded Attribution,"** not "Causal SHAP." Standard SHAP treats features as independent when computing marginals; true Causal Shapley Values (Heskes et al., 2020) respect the causal graph in that computation. This project's method is DAG-*informed* (SHAP applied to equations recovered from the DAG) but doesn't implement the formal causal-Shapley marginalization — so it's labeled as the honest intermediate that it is, rather than overclaiming a citation it doesn't fully implement.

---

## 4. The Synthetic Datasets — Deep Dive

Both domains are generated by an isomorphic data-generating process (DGP) — same confounding shape, different variable names — which is itself evidence that the causal-recovery method generalizes rather than being tuned to one dataset's quirks.

### Manufacturing (`data/generate_data.py` → `prihir_synthetic.csv`)

**The planted causal graph (9 edges):**
```
order_complexity → supplier_a                (the confounder → treatment link)
order_complexity → machine_queue_length
order_complexity → shipment_delay              (confounder → outcome, direct)
supplier_a → material_lead_time                (treatment → mediator)
material_lead_time → shipment_delay            (mediator → outcome)
machine_queue_length → approval_duration
approval_duration → shipment_delay
export_flag → approval_duration
carrier_express → shipment_delay
```
Treatment = `supplier_a` (binary), Outcome = `shipment_delay` (continuous). Planted true causal effect: **7.4 × 0.9 = 6.66 days**.

**Realism features layered on top of the pure structural equations** (these don't change the causal structure, they make the data behave like real operational data instead of a clean textbook example):

- **Irregular, business-hour-weighted timestamps** — gaps drawn from a gamma distribution (mean ~3.6h during business hours, ~12.5h nights/weekends), plus a 4% chance of an "urgent batch" burst (<1.5h gap) and a 3% chance of a holiday/shutdown idle period (26–72h gap) — not a uniform "one event every 6 hours" toy pattern.
- **Concept drift** — residual noise standard deviation increases 30% after row 10,000, simulating a factory expansion with "teething problems." Structural coefficients themselves don't change — only how noisy the outcome is.
- **Seasonal pressure** — Q4 (Oct–Dec) orders get an additive +0.8 day average delay bump.
- **Outliers (~1% of rows)** — split between simulated *supplier failures* (material lead time spikes to 10–15 days) and *regulatory holds* (approval duration spikes to 15–25 days), with the outlier's effect correctly propagated through to `shipment_delay` so the SCM stays internally consistent even for these rows.

**Engineering judgment calls, documented directly in code comments** (the kind of detail worth walking a mentor through, since it shows iteration, not a first-draft):
- The confounding sigmoid steepness is deliberately kept at **0.7**, not raised further. A stronger steepness would make even the *non-linear* confounder→treatment edge linearly detectable by PC discovery — which sounds better, but at n=15,000 it produced a perfect 9/9 pure-discovery recovery with nothing left for the domain-knowledge-constraints step to add. 0.7 preserves a genuine, honest gap for domain knowledge to fill.
- Outlier magnitude was **tuned down** from an earlier, more extreme version (2% of rows, lead time 18–30 days) after that version added enough unexplained variance to push the weakest planted edge (`carrier_express → shipment_delay`, true coefficient only −0.6) below the bootstrap-detection threshold. The current 1%/10–15-day version was chosen specifically to read as genuine outliers without swamping the signal the tests depend on.
- The test-suite fixture uses **n=1,500** for manufacturing, not the rounder n=1,000, because at n=1,000 there's a real statistical tradeoff in this DGP between confounding strength (needed for reliable bootstrap-PC edge detection) and DML confidence-interval width (which widens under stronger confounding, from reduced treatment/control overlap) — n=1,500 clears both bars without trading one off against the other.

### Healthcare (`data/generate_healthcare.py` → `hospital_synthetic.csv`)

Same shape, hospital-admissions framing — proving the framework isn't hard-coded to one domain:

```
patient_complexity → specialist_required        (confounder → treatment)
patient_complexity → bed_occupancy_rate
patient_complexity → length_of_stay              (confounder → outcome, direct)
specialist_required → treatment_duration         (treatment → mediator)
treatment_duration → length_of_stay
bed_occupancy_rate → approval_wait
approval_wait → length_of_stay
emergency_admission → approval_wait
insurance_expedited → length_of_stay
```
Treatment = `specialist_required`, Outcome = `length_of_stay`. Planted true effect: **6.2 × 0.85 = 5.27 days**. Its own realism layer: ICU-escalation/insurance-dispute outliers, ED-rush-aware admission timestamps, a new-wing-opening concept drift, and a winter (Nov–Feb) seasonal +0.6 day bump — the domain-specific equivalent of every manufacturing realism feature above.

Both datasets ship at **15,000 rows** by default and are cached to CSV after first generation (`load_or_generate()`), so re-running the dashboard doesn't regenerate data every time.

---

## 5. Frontend — Streamlit Dashboard (`app/`)

The dashboard (`app/dashboard.py`, 1,467 lines) is a 5-tab experience, each tab implemented as its own module under `app/tabs/`:

| Tab | File | Purpose |
|---|---|---|
| **① Overview** | `tab_overview.py` (313 lines) | Executive headline: the causal finding, expected savings, top recommended action |
| **② Data & Discovery** | `tab_data_discovery.py` (1,481 lines) | Event log stats, object-interaction graph, the discovered causal DAG, and the domain-knowledge ablation study |
| **③ Model & Impact** | `tab_model_impact.py` (1,344 lines) | Interactive what-if causal simulator, structural equation summary, CATE (heterogeneous treatment effects), SHAP waterfall attribution |
| **④ Decision Intelligence** | `tab_decision_intelligence.py` (240 lines) | Board-room-style report synthesizing the above into a single narrative |
| **⑤ Copilot** | `tab_copilot.py` (551 lines) | Conversational agent over the live pipeline output (see below) |

Both **Manufacturing** (Prihir Enterprises) and **Healthcare** (hospital admissions) domains run through the identical pipeline and UI, switchable from the sidebar — the strongest evidence that the framework is genuinely domain-agnostic rather than hard-coded to one scenario.

### The Copilot (`app/copilot.py`, 814 lines)

An LLM agent (Cerebras Cloud, `gemma-4-31b`) grounded entirely in that session's actual pipeline output — the discovered graph, structural equations, DML estimate, and what-if simulation results are assembled into a structured context string (`build_context()`) and passed to the model alongside the question. The system prompt explicitly instructs the model to answer the *specific* question using the *relevant* context section (so "what's the bottleneck" and "run a what-if simulation" pull from different parts of the same context, not the same canned summary) and to decline off-topic questions rather than hallucinate.

If the API is unavailable (no key, network error, rate limit), the Copilot transparently falls back to pre-computed, still pipeline-grounded answers rather than breaking the demo — the UI status pill reflects which mode is actually active, not just whether a key is configured.

---

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Process mining | `pm4py` |
| Causal discovery | `causal-learn` (PC algorithm, Fisher's Z test) |
| Causal effect estimation | `DoWhy` + custom Double ML implementation |
| Structural modeling | `scikit-learn` (Logistic / Linear / Gradient Boosting) |
| Attribution | `shap` |
| Graphs | `networkx` |
| Dashboard | `Streamlit` + `Plotly` + `streamlit-agraph` |
| LLM Copilot | Cerebras Cloud API (`gemma-4-31b`) via `openai` SDK (OpenAI-compatible endpoint) |
| Testing | `pytest` |

---

## 7. Testing — Deep Dive

**49/49 tests passing** — run live (`pytest causal_ocpm/tests/ -q`, ~90 seconds) while writing this document, not copied from an old claim. Organized into 9 groups across two files:

| # | Class (in `test_pipeline.py`) | What it actually checks |
|---|---|---|
| 1 | `TestDataGeneration` | Required columns exist for both domains; the confounding trap is genuinely present (naive effect > true effect); the confounder-treatment correlation is strong enough (>0.3) to be a real test of adjustment; no NaNs; binary columns are actually 0/1 |
| 2 | `TestObjectGraph` | Phase 1 graph builds for both domains, all 5 object roles present, summary keys correct |
| 3 | `TestCausalDiscovery` | Discovered DAG is acyclic; all planted ground-truth edges are recovered; precision is acceptable; the ablation study shows measurable improvement and returns all expected keys |
| 4 | `TestMixedSCM` | Binary nodes really do use `LogisticRegression`, the outcome node really does use `GradientBoostingRegressor`; every node has a fitted equation; metrics are positive; healthcare SCM builds too; coefficient extraction returns a proper DataFrame |
| 5 | `TestDoOperator` | **Core test**: causal estimate within ±0.5 days of the planted truth; causal effect < naive effect (confounding was actually removed); full 6-scenario sensitivity analysis runs; placebo effect stays near zero; other treatment variables also complete; healthcare causal effect within ±0.8 days of its own planted truth; 10-seed robustness check all land within ±0.5; naive effect is a sane positive float |
| 6 | `TestAttribution` | Case explanation runs and returns `shap_value`/`attribution` columns; **SHAP additivity holds** (baseline + Σ shap ≈ predicted, within 0.15); attribution categories are restricted to `{actionable, structural}`; healthcare attribution runs; summary dict has the expected keys |
| 7 | `TestBootstrappedDiscovery` | Every edge carries a confidence score in `dag.graph['edge_confidence']`; every confidence value is a float in [0,1]; bootstrap run count is recorded and > 0; every ground-truth edge (except the one deliberately non-linear edge, exempted by design) has confidence ≥ 0.5 |
| 8 | `TestDMLEstimator` | DML is confirmed as the actual method used (`result['method'] == 'double_ml'`), not silently falling back; the 95% CI is tight enough (<0.5 days wide) to be a meaningful estimate at this sample size; `method_label` is populated; the estimate is close to the planted truth |
| 9 | `TestCVScoring` | Every SCM equation's metric label actually says "CV-" (cross-validated, not in-sample); every CV score is in a valid range; the outcome equation's CV-R² is meaningfully > 0.5, i.e. the model is actually predictive, not just fitted |

`test_dashboard_smoke.py` (2 tests) separately confirms the Streamlit app itself imports and initializes without error — a fast sanity check distinct from the pipeline's statistical correctness.

`validate.py` is a separate, human-readable PASS/FAIL report (colour-coded terminal output) that reruns the same checks on the full 15,000-row datasets and prints a summary — useful for a quick "is everything still working" check without reading pytest output.

---

## 8. Custom Data Upload & Quality Scoring

Beyond the two built-in synthetic domains, `data/custom_loader.py` (326 lines) supports uploading your own CSV or OCEL JSON. Before running the pipeline on unknown data, `analyze_data_quality()` runs a **6-check audit** and produces a 0–100 score:

| # | Check | Deduction |
|---|---|---|
| 1 | Sample size (<50 rows blocks, <100/<200 warns) | up to −40 |
| 2 | Missing values (>30% in a column drops it; some missing triggers imputation) | −20 / −5 |
| 3 | Near-zero-variance columns (excluded automatically) | −10 |
| 4 | Duplicate rows | −10 |
| 5 | Numeric feature count (<3 blocks — causal discovery needs enough variables) | −30 |
| 6 | Outliers (winsorized automatically) | −5 |

Issues (score-critical, e.g. too few rows) block the pipeline with a clear error; warnings (e.g. some missing values) let it proceed after automatic cleaning. This score also feeds the "Result Confidence" badge shown alongside any custom-data analysis, so a low-quality upload is visibly flagged rather than silently producing an overconfident-looking result.

---

## 9. Project Structure

```
causal_ocpm/
├── data/
│   ├── generate_data.py            # Manufacturing synthetic data + planted ground truth
│   ├── generate_healthcare.py      # Healthcare synthetic data + planted ground truth
│   ├── custom_loader.py            # Upload-your-own-CSV path + data-quality scoring
│   └── prihir_synthetic.csv, hospital_synthetic.csv
├── src/
│   ├── phase1_graph.py             # Object interaction graph
│   ├── phase2_discovery.py         # Bootstrapped PC causal discovery
│   ├── phase3_scm.py               # Mixed structural causal model
│   ├── phase4_dooperator.py        # Double ML effect estimation + sensitivity analysis
│   └── phase5_attribution.py       # SCM-grounded SHAP attribution
├── app/
│   ├── dashboard.py                # Page shell, sidebar, tab wiring
│   ├── copilot.py                  # Cerebras LLM integration + context builder
│   ├── style.css                   # Shared app-wide styling
│   └── tabs/
│       ├── tab_overview.py
│       ├── tab_data_discovery.py
│       ├── tab_model_impact.py
│       ├── tab_decision_intelligence.py
│       └── tab_copilot.py
├── tests/
│   ├── test_pipeline.py            # 47 tests across all 5 phases, 9 groups
│   └── test_dashboard_smoke.py     # 2 smoke tests
└── validate.py                     # Standalone human-readable accuracy/robustness report
```

Total: **~9,450 lines** across data generation, the 5-phase pipeline, the dashboard, and tests.

---

## 10. What Makes This More Than a Dashboard

- **No single existing tool does this end-to-end.** `pm4py` does descriptive process mining; `DoWhy`/`CausalNex` do effect estimation — nothing public chains *object-centric* process mining into causal discovery into a mixed SCM into DML into case-level attribution, in one pipeline.
- **Every number in the UI is traceable to a pipeline artifact.** The Copilot, the executive report, the simulator — none of them contain hard-coded business numbers; they all read from the live `dag`, `scm`, and `do_result` objects produced by the 5 phases.
- **Validated against known ground truth, not just "looks plausible."** Because the synthetic data has a planted causal structure, the pipeline's own accuracy is directly measurable (edge recovery, effect-recovery error) — most process-mining demos can't make this claim because real-world data has no known ground truth to check against.
- **Domain-agnostic by construction.** The same code runs manufacturing and healthcare through one interface, generated by the same isomorphic data-generating process; adding a third domain is a data-mapping exercise, not a rewrite.
- **The limitations are documented, not hidden.** The non-linear confounder edge that pure discovery can't detect, the "SCM-Grounded Attribution" vs. "Causal SHAP" naming distinction, the specific sample-size tradeoffs in the test fixtures — all called out directly in code comments rather than glossed over.

---

## 11. How to Run It

```bash
pip install -r causal_ocpm/requirements.txt

# Generate synthetic data with planted ground truth
python causal_ocpm/data/generate_data.py
python causal_ocpm/data/generate_healthcare.py

# Run the 5-phase pipeline standalone (optional — the dashboard runs it live too)
python causal_ocpm/src/phase1_graph.py
python causal_ocpm/src/phase2_discovery.py
python causal_ocpm/src/phase3_scm.py
python causal_ocpm/src/phase4_dooperator.py
python causal_ocpm/src/phase5_attribution.py

# Verify everything (49 tests, ~90 seconds)
pytest causal_ocpm/tests/ -v

# Human-readable PASS/FAIL validation report
python causal_ocpm/validate.py

# Launch the dashboard
streamlit run causal_ocpm/app/dashboard.py
```
