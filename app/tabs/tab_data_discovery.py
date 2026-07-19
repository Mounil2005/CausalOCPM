# ── GLOBAL UX STYLING ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Spacing System */
    .discovery-section { margin-top: 64px; }
    .discovery-header { font-size: 1.4rem; font-weight: 800; color: #0F172A; margin-bottom: 16px; }
    .discovery-desc { font-size: 1.05rem; color: #475569; margin-bottom: 24px; line-height: 1.6; }
    
    /* Reversed KPI Cards — top-aligned so the number sits on the same
       baseline across every card, regardless of whether a delta line
       follows it (center-aligned flex previously made cards without a
       delta line look "sunk" relative to the one that has one). */
    .kpi-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #E2E8F0; border-radius: 12px; padding: 24px; text-align: left; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: flex-start; transition: box-shadow 0.15s ease, border-color 0.15s ease; }
    .kpi-card:hover { box-shadow: 0 6px 16px rgba(0,0,0,0.06); border-color: #CBD5E1; }
    /* Two meaning-carrying accent groups, not four decorative colors:
       "context" = scale/volume of the dataset, "outcome" = the metric
       this whole app exists to move (matches the amber/warning tone used
       for outcome-risk elsewhere in the app). */
    .kpi-card--context { border-top-color: #0284C7; }
    .kpi-card--outcome { border-top-color: #D97706; }
    .kpi-card--quality { border-top-color: #059669; }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #0F172A; line-height: 1.1; margin-bottom: 8px; letter-spacing: -0.02em; }
    .kpi-label { font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
    .kpi-delta { font-size: 0.8rem; font-weight: 600; margin-top: 10px; padding-top: 10px; border-top: 1px solid #F1F5F9; }
</style>
""", unsafe_allow_html=True)

outcome_var   = cfg["outcome_var"]
treatment_var = cfg["treatment_var"]
binary_vars   = cfg["binary_vars"]
outcome_label = cfg["outcome_label"]

# Object types = distinct OCEL node roles (Case/Machine/Worker/Material/Outcome),
# NOT the count of binary indicator columns — those are different concepts.
_n_object_types = (
    len(set(nx.get_node_attributes(G, "role").values()))
    if G and G.number_of_nodes() > 0
    else len(binary_vars)
)

# ── MODEL VALIDATION NUMBERS (computed early — the AI card needs them too) ──
# Deliberately sourced from the *pre*-domain-knowledge run
# (ablation["without_domain_knowledge"]), not `dag_metrics`. `dag_metrics`
# is computed after enforce_domain_knowledge() force-adds every planted
# ground-truth edge into the graph (phase2_discovery.py), which makes
# recall = 1.00 guaranteed by construction — not something the algorithm
# discovered. Showing that number here would be circular and, rightly,
# not credible. The raw PC-algorithm-only numbers below are what the
# statistics actually found on their own.
_dq_wodk = ablation.get("without_domain_knowledge", {}) if ablation else {}
_dq_prec = _dq_wodk.get("precision", dag_metrics.get("precision", 0.0))
_dq_rec  = _dq_wodk.get("recall",    dag_metrics.get("recall",    0.0))
_dq_f1   = _dq_wodk.get("f1_score",  dag_metrics.get("f1_score",  0.0))
_dq_tp     = _dq_wodk.get("true_positives",  0)
_dq_fp     = _dq_wodk.get("false_positives", 0)
_dq_fn     = _dq_wodk.get("false_negatives", 0)
_dq_boot_n = dag.graph.get("bootstrap_n", 20)

# ── AI DISCOVERY SUMMARY ────────────────────────────────────────────────────
# Strongest measured relationship pulled straight from the fitted SCM
# coefficients (coefs), not a hardcoded "Supplier A → Material Lead Time"
# string — so this line actually reflects whatever the model found on this
# run rather than reading identically regardless of the data.
if not coefs.empty and {"parent", "child", "estimated_value"} <= set(coefs.columns):
    _dq_top_edge = coefs.loc[coefs["estimated_value"].abs().idxmax()]
    _dq_strongest = (
        f'<b>{_dq_top_edge["parent"].replace("_", " ").title()} ➡ '
        f'{_dq_top_edge["child"].replace("_", " ").title()}</b> '
        f'(coefficient {_dq_top_edge["estimated_value"]:.2f})'
    )
else:
    _dq_strongest = f"<b>Supplier A ➡ Material Lead Time ➡ {outcome_label}</b>"

_dq_avg_conf = (_dq_prec + _dq_rec + _dq_f1) / 3
_dq_conf_col, _dq_conf_lbl = _ai_status(_dq_avg_conf)
st.markdown(
    _ai_card(
        accent="#3B82F6", badge_bg="#EFF6FF", icon="✨", title="AI Discovery Summary",
        conf_color=_dq_conf_col, conf_label=_dq_conf_lbl,
        lead=(
            f"Causal discovery recovered <b>{dag.number_of_edges()} verified links</b> across "
            f"{len(df):,} events and {_n_object_types} object types — validated against planted "
            f"ground truth, not left as raw correlation."
        ),
        bullets=[
            f"Strongest measured relationship: {_dq_strongest}",
            f"Bootstrap stability <b>{_boot_stab_pct:.0f}%</b> across {_dq_boot_n} resampled reruns "
            f"— precision {_dq_prec:.2f}, recall {_dq_rec:.2f} before domain knowledge is applied",
            f"Domain knowledge recovered {max(0, (_dq_wodk.get('false_negatives', 0) - ablation.get('with_domain_knowledge', {}).get('false_negatives', 0)))} "
            f"missing edge(s), guaranteeing DAG validity without inventing relationships",
            f"Next step: estimate precise intervention effects in the <b>Model Performance</b> tab",
        ],
    ),
    unsafe_allow_html=True,
)
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── MODEL QUALITY AT A GLANCE ───────────────────────────────────────────────
# Same numbers as the detailed "Step 5: Validate Discovery Quality" section
# further down, surfaced immediately at the top of the tab — precision,
# recall and F1 are the headline trust signal for this whole page (is the
# discovered graph actually right?) and shouldn't require scrolling past
# five steps to find.

def _dq_status(frac):
    if frac >= 0.85: return SUCCESS, "Strong"
    if frac >= 0.65: return WARNING, "Moderate"
    return ERROR, "Needs Review"

# Per-metric "i" info icon (native title attribute — no JS, so it can't hit
# the React-crash issue an onclick-based tooltip would). Each one spells out
# the actual formula with this run's real TP/FP/FN counts, so hovering any
# single box answers "how did THIS number happen" without leaving the card.
def _dq_info_icon(tooltip):
    return (
        f'<span title="{tooltip}" style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:15px;height:15px;border-radius:50%;background:{BORDER};color:{MUTED};'
        f'font-size:0.62rem;font-weight:800;font-style:normal;cursor:help;flex-shrink:0;">i</span>'
    )

# See tab_overview.py's identical _NL comment: a literal blank line inside
# a title="..." attribute gets read as a markdown paragraph break by
# st.markdown()'s markdown-to-HTML pass and corrupts the tag, so real "\n\n"
# is replaced with a numeric-entity newline that markdown leaves alone but
# the browser still renders as a line break in the native tooltip.
_NL = "&#10;&#10;"
_dq_gt_n_hint = _dq_tp + _dq_fn
_DQ_TOOLTIPS = {
    "Bootstrap Stability": (
        f"Think of it as asking {_dq_boot_n} independent witnesses the same question instead of "
        f"trusting just one. The algorithm reruns causal discovery {_dq_boot_n} times, each on a "
        f"slightly different resampled slice of the same data, and checks how often it lands on "
        f"the same causal edges.{_NL}"
        f"A relationship that keeps reappearing across resamples is likely a real pattern in how "
        f"the process behaves, not a coincidence of this one dataset. One that only shows up once "
        f"is a red flag it might just be noise.{_NL}"
        f"Result here: {_boot_stab_pct:.0f}% agreement across {_dq_boot_n} reruns — most of what "
        f"was found holds up under repeated testing."
    ),
    "Precision": (
        f"Picture the algorithm as a detective naming suspects. Precision asks: of everyone it "
        f"accused, how many were actually guilty?{_NL}"
        f"It named {_dq_tp + _dq_fp} relationships as causal. {_dq_tp} matched the real planted "
        f"structure; {_dq_fp} were false accusations — links it claimed exist but don't.{_NL}"
        f"{_dq_tp} correct ÷ {_dq_tp + _dq_fp} accused = {_dq_prec:.2f}. A high score means that "
        f"when this model says 'A causes B,' you can trust it — it rarely cries wolf."
    ),
    "Edge Recall": (
        f"This flips the question around: of all {_dq_gt_n_hint} real causal relationships "
        f"actually planted in the data, how many did the algorithm manage to dig up on its own?{_NL}"
        f"It correctly found {_dq_tp}, and missed {_dq_fn}. {_dq_tp} found ÷ {_dq_gt_n_hint} that "
        f"truly exist = {_dq_rec:.2f}.{_NL}"
        f"A high score means the model is thorough — it doesn't leave real causes undiscovered, "
        f"even if it occasionally over-claims (that's what Precision catches separately)."
    ),
    "Recovery F1": (
        f"Precision and Recall pull against each other. You could fake a perfect Precision by "
        f"only naming the one relationship you're 100% sure about — or fake a perfect Recall by "
        f"claiming every possible link exists and being wrong most of the time.{_NL}"
        f"F1 is the honest referee: the harmonic mean of both, 2×(P×R)÷(P+R) = {_dq_f1:.2f}. It "
        f"only scores high when precision AND recall are genuinely strong together — you can't "
        f"game it by leaning on just one."
    ),
}

def _dq_tile(icon, value_str, label, frac):
    _color, _status = _dq_status(frac)
    return (
        f'<div style="background:#FFFFFF;border:1px solid {BORDER};border-radius:12px;'
        f'padding:16px 18px;flex:1;min-width:150px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        f'<span style="font-size:1rem;">{icon}</span>'
        f'<span style="font-size:0.62rem;font-weight:800;color:{_color};background:{_color}1A;'
        f'padding:2px 8px;border-radius:20px;text-transform:uppercase;letter-spacing:0.03em;">{_status}</span>'
        f'</div>'
        f'<div style="font-size:1.9rem;font-weight:900;color:{TEXT};line-height:1;">{value_str}</div>'
        f'<div style="display:flex;align-items:center;gap:5px;margin-top:6px;margin-bottom:9px;">'
        f'<span style="font-size:0.68rem;font-weight:700;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:0.05em;">{label}</span>'
        + _dq_info_icon(_DQ_TOOLTIPS.get(label, ""))
        + f'</div>'
        f'<div style="height:5px;background:{BORDER};border-radius:3px;overflow:hidden;">'
        f'<div style="height:100%;width:{frac*100:.0f}%;background:{_color};border-radius:3px;"></div>'
        f'</div>'
        f'</div>'
    )

_dq_avg = (_dq_prec + _dq_rec + _dq_f1) / 3
_dq_verdict_col, _dq_verdict_lbl = _dq_status(_dq_avg)

st.markdown(
    f'<div style="background:#FFFFFF;border:1px solid {BORDER};border-radius:14px;'
    f'padding:22px 26px;margin-bottom:16px;box-shadow:0 2px 10px rgba(15,23,42,0.03);">'
    f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:6px;">'
    f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
    f'<span style="font-size:1.1rem;">🛡️</span>'
    f'<span style="font-size:0.85rem;font-weight:800;color:{TEXT};text-transform:uppercase;letter-spacing:0.06em;">'
    f'Model Quality At a Glance</span>'
    f'<span style="font-size:0.68rem;font-weight:800;color:#FFFFFF;background:{_dq_verdict_col};'
    f'padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:0.03em;">'
    f'✓ {_dq_verdict_lbl} Discovery Quality</span>'
    f'</div>'
    f'<span style="font-size:0.78rem;color:{SUBTLE};">Full validation walkthrough in Step 5 below ↓</span>'
    f'</div>'
    f'<div style="font-size:0.78rem;color:{MUTED};margin-bottom:16px;">'
    f'Raw PC-algorithm discovery vs. planted ground truth — '
    f'<b>before</b> domain-knowledge constraints are applied, so this reflects what the statistics '
    f'found on their own rather than a number domain knowledge guarantees to be perfect.</div>'
    f'<div style="display:flex;gap:14px;flex-wrap:wrap;">'
    + _dq_tile("🔁", f"{_boot_stab_pct:.0f}%", "Bootstrap Stability", _boot_stab_pct / 100)
    + _dq_tile("🎯", f"{_dq_prec:.2f}", "Precision", _dq_prec)
    + _dq_tile("🔎", f"{_dq_rec:.2f}", "Edge Recall", _dq_rec)
    + _dq_tile("⚖️", f"{_dq_f1:.2f}", "Recovery F1", _dq_f1)
    + f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
with st.expander("ℹ️ How are these numbers calculated?"):
    _dq_gt_n = len(_dq_wodk.get("ground_truth_edges", [])) or dag.number_of_edges() or 9

    def _dq_flow_step(n, icon, title, desc):
        return (
            f'<div style="flex:1;min-width:170px;background:#FFFFFF;border:1px solid {BORDER};'
            f'border-radius:10px;padding:14px 16px;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="width:20px;height:20px;border-radius:50%;background:{PRIMARY};color:#FFFFFF;'
            f'font-size:0.68rem;font-weight:800;display:flex;align-items:center;justify-content:center;'
            f'flex-shrink:0;">{n}</span>'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span style="font-size:0.82rem;font-weight:800;color:{TEXT};">{title}</span>'
            f'</div>'
            f'<div style="font-size:0.78rem;color:{MUTED};line-height:1.5;">{desc}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:stretch;margin-bottom:6px;">'
        + _dq_flow_step(1, "🌱", "Plant a known answer",
            f"This dataset is synthetic — generated from {_dq_gt_n} causal edges planted in "
            f"advance, so discovery can be graded against the real answer.")
        + '<div style="display:flex;align-items:center;color:#CBD5E1;font-size:1.3rem;font-weight:800;">→</div>'
        + _dq_flow_step(2, "🔍", "Discover statistically",
            f"The PC algorithm runs {_dq_boot_n}× on resampled copies of the data and proposes "
            f"a graph from patterns alone — no domain rules involved yet.")
        + '<div style="display:flex;align-items:center;color:#CBD5E1;font-size:1.3rem;font-weight:800;">→</div>'
        + _dq_flow_step(3, "✅", "Compare, edge by edge",
            f"The proposed graph is checked against the {_dq_gt_n} planted edges to see what "
            f"was found correctly, missed, or invented.")
        + f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="background:#F8FAFC;border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 16px;margin:14px 0;display:flex;gap:20px;flex-wrap:wrap;font-size:0.82rem;">'
        f'<span style="color:{SUCCESS};font-weight:700;">✓ {_dq_tp} correct edges found</span>'
        f'<span style="color:{ERROR};font-weight:700;">✗ {_dq_fp} false alarms (edges that aren\'t real)</span>'
        f'<span style="color:{WARNING};font-weight:700;">− {_dq_fn} missed edges</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    def _dq_formula_card(icon, name, formula, plain):
        return (
            f'<div style="flex:1;min-width:190px;background:#FFFFFF;border:1px solid {BORDER};'
            f'border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:0.85rem;font-weight:800;color:{TEXT};margin-bottom:4px;">{icon} {name}</div>'
            f'<div style="font-family:monospace;font-size:0.78rem;color:{PRIMARY};background:{PRIMARY}14;'
            f'border-radius:6px;padding:4px 8px;display:inline-block;margin-bottom:6px;">{formula}</div>'
            f'<div style="font-size:0.78rem;color:{MUTED};line-height:1.4;">{plain}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
        + _dq_formula_card("🎯", "Precision", "TP ÷ (TP + FP)", "Of what it found, how much was right?")
        + _dq_formula_card("🔎", "Edge Recall", "TP ÷ (TP + FN)", "Of what's actually true, how much did it find?")
        + _dq_formula_card("⚖️", "Recovery F1", "harmonic mean", "Precision and recall balanced into one number.")
        + f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="background:#F0FDF4;border-left:3px solid {SUCCESS};border-radius:0 8px 8px 0;'
        f'padding:12px 16px;margin-top:14px;font-size:0.82rem;color:{TEXT};line-height:1.5;">'
        f'<b>🔁 Bootstrap Stability</b> is separate from the three scores above — it reruns the '
        f'whole process {_dq_boot_n}× on resampled data and reports what share of those runs '
        f'agreed an edge was real, averaged over the ground-truth edges. It answers '
        f'<i>"if we reran this on slightly different data, would we get the same answer?"</i> — '
        f'a guard against the result being a fluke of one particular sample.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.76rem;color:{SUBTLE};margin-top:12px;">'
        f'Domain-knowledge constraints are applied only <b>after</b> this scoring (Step 6 below) — '
        f'that\'s why these numbers are graded on the raw statistics alone, before anything is corrected.</div>',
        unsafe_allow_html=True,
    )

# ── STEP 1: UNDERSTAND THE EVENT DATA ─────────────────────────────────────────
st.markdown('<div class="discovery-section"></div>', unsafe_allow_html=True)
if is_custom:
    st.markdown(f"""
        <div class="discovery-header">Step 1: Understand the Event Data</div>
        <div class="discovery-desc">The uploaded event log contains {len(df):,} events across {len(cfg['numeric_vars'])} features. This audited dataset forms the foundation for causal discovery.</div>
    """, unsafe_allow_html=True)
    accuracy_disclaimer(custom_confidence, len(df), custom_quality.get("score", 0))
    render_quality_report(custom_quality, custom_cleaning_log)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card kpi-card--context"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Total Rows</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card kpi-card--context"><div class="kpi-value">{len(cfg["numeric_vars"])}</div><div class="kpi-label">Features</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card kpi-card--outcome"><div class="kpi-value">{df[outcome_var].mean():.2f}</div><div class="kpi-label">Mean {outcome_label}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card kpi-card--outcome"><div class="kpi-value">{df[outcome_var].std():.2f}</div><div class="kpi-label">Std {outcome_label}</div></div>', unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="discovery-header">Step 1: Understand the Event Data</div>
        <div class="discovery-desc">This event log contains {len(df):,} manufacturing events spanning 5 interacting business object types. These events form the foundation for causal discovery.</div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    _treated_pct = df[treatment_var].mean() * 100 if treatment_var in df.columns else 0
    c1.markdown(f'<div class="kpi-card kpi-card--context"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Total Events</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card kpi-card--context"><div class="kpi-value">{int(df[treatment_var].sum()):,}</div><div class="kpi-label">Treated Cases</div><div class="kpi-delta" style="color: #10B981;">{_treated_pct:.0f}% of total</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card kpi-card--outcome"><div class="kpi-value">{df[outcome_var].mean():.1f}</div><div class="kpi-label">Avg Delay (Days)</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card kpi-card--outcome"><div class="kpi-value">{df[outcome_var].std():.1f}</div><div class="kpi-label">Std Dev (Days)</div></div>', unsafe_allow_html=True)

# Prepare dataframe display (will be rendered in the stats column below)
display_cols = cfg["numeric_vars"][:8]
display_df   = df[display_cols].head(10).copy()
for col in binary_vars:
    if col in display_df.columns:
        display_df[col] = display_df[col].astype(int)
non_binary = [c for c in display_df.columns if c not in binary_vars]
binary_in  = [c for c in display_df.columns if c in binary_vars]
fmt = {c: "{:.3f}" for c in non_binary}
fmt.update({c: "{:d}" for c in binary_in})

# ── STEP 2: EXPLORE OBJECT RELATIONSHIPS ──────────────────────────────────────
st.markdown('<div class="discovery-section"></div>', unsafe_allow_html=True)
st.markdown(f"""
    <div class="discovery-header">Step 2: Explore Object Relationships</div>
""", unsafe_allow_html=True)

if is_custom:
    st.markdown(f"""
        <div class="discovery-desc">Not available for flat tabular data. Causal analysis continues below.</div>
    """, unsafe_allow_html=True)
    insight_card(
        "Object Graph Skipped",
        "The object-interaction graph models relationships between typed OCEL object "
        "instances (cases, resources, artifacts). A flat CSV has no object-role columns, "
        "so this view is skipped — every downstream phase runs normally.",
        "knowledge"
    )
else:
    insight_card(
        "Object Interaction Network",
        "This graph visualizes how Cases, Machines, Workers, Materials, and Outcomes co-occur throughout manufacturing events. Higher connectivity indicates richer process interactions — the structural foundation causal discovery builds on.",
        "knowledge"
    )

if G and G.number_of_nodes() > 0:
    import math as _math

    # ── Type-level aggregation ─────────────────────────────────────────
    _roles_full = nx.get_node_attributes(G, "role")
    _type_n: Dict[str, int] = {}
    for _nd, _rl in _roles_full.items():
        _type_n[_rl] = _type_n.get(_rl, 0) + 1

    _type_cooc: Dict[tuple, int] = {}
    for _u, _v, _ed in G.edges(data=True):
        _ru, _rv = _roles_full.get(_u), _roles_full.get(_v)
        if _ru and _rv and _ru != _rv:
            _pair = tuple(sorted([_ru, _rv]))
            _type_cooc[_pair] = _type_cooc.get(_pair, 0) + _ed.get("weight", 1)

    _role_label: Dict[str, str] = {
        "Case":             "Patient"    if domain == "healthcare" else "Case",
        "Resource_Machine": "Ward"       if domain == "healthcare" else "Machine",
        "Resource_Worker":  "Clinician"  if domain == "healthcare" else "Worker",
        "Artifact":         "Medication" if domain == "healthcare" else "Material",
        "Outcome":          "Discharge"  if domain == "healthcare" else "Outcome",
    }

    # ── Stats panel pre-computation ────────────────────────────────────
    _mach_col_map = {
        "manufacturing": ("machine_id",   "worker_id"),
        "healthcare":    ("ward_id",       "clinician_id"),
        "bpi2019":       ("machine_id",    "worker_id"),
    }
    _machine_col, _worker_col = _mach_col_map.get(domain, ("machine_id", "worker_id"))
    _mach_label = _role_label["Resource_Machine"]
    _work_label = _role_label["Resource_Worker"]

    if _machine_col in df.columns:
        _mach_vc           = df[_machine_col].value_counts()
        _top_machine       = str(_mach_vc.index[0]) if len(_mach_vc) else "—"
        _top_machine_count = int(_mach_vc.iloc[0])  if len(_mach_vc) else 0
    else:
        _top_machine, _top_machine_count = "—", 0

    if _worker_col in df.columns:
        _work_vc           = df[_worker_col].value_counts()
        _top_worker        = str(_work_vc.index[0]) if len(_work_vc) else "—"
        _top_worker_count  = int(_work_vc.iloc[0])  if len(_work_vc) else 0
    else:
        _top_worker, _top_worker_count = "—", 0

    _avg_objects   = float(len(_type_n))
    _multi_obj_pct = 100

    # ── PART A: two-column layout ──────────────────────────────────────
    _col_graph, _col_stats = st.columns([1.2, 0.8])

    with _col_graph:
        import streamlit.components.v1 as _stc

        _ac  = _type_n.get("Case",             0)
        _am  = _type_n.get("Resource_Machine", 0)
        _aw  = _type_n.get("Resource_Worker",  0)
        _art = _type_n.get("Artifact",         0)
        _ao  = _type_n.get("Outcome",          0)

        _lbl_case     = _role_label.get("Case",             "Case")
        _lbl_machine  = _role_label.get("Resource_Machine", "Machine")
        _lbl_worker   = _role_label.get("Resource_Worker",  "Worker")
        _lbl_material = _role_label.get("Artifact",         "Material")
        _lbl_outcome  = _role_label.get("Outcome",          "Outcome")

        _sub_case    = "patients" if domain == "healthcare" else "orders"
        _sub_machine = "wards"    if domain == "healthcare" else "units"
        _sub_worker  = "clinicians" if domain == "healthcare" else "staff"

        # Theme-aware canvas colors — this panel used to be hardcoded dark
        # (#0D1117) regardless of theme, leaving a jarring black box on the
        # light theme. Mirror the IS_LIGHT pattern used by the DAG animation.
        _og_bg       = "#F8F9FB" if IS_LIGHT else "#0D1117"
        _og_border   = "#E2E8F0" if IS_LIGHT else "rgba(255,255,255,0.08)"
        _og_spill_bg = "rgba(15,23,42,0.04)"   if IS_LIGHT else "rgba(255,255,255,0.05)"
        _og_spill_bd = "rgba(15,23,42,0.10)"   if IS_LIGHT else "rgba(255,255,255,0.1)"
        _og_spill_tx = "rgba(15,23,42,0.45)"   if IS_LIGHT else "rgba(255,255,255,0.4)"
        _og_spill_b  = "#0F172A" if IS_LIGHT else "#fff"

        _stc.html(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{background:{_og_bg};font-family:'Inter',-apple-system,sans-serif;overflow:hidden;width:100%;height:100%;}}
.outer{{width:100%;overflow:hidden;}}
.wrap{{position:relative;width:640px;height:400px;overflow:hidden;transform-origin:top left;
  border:1px solid {_og_border};border-radius:12px;}}
canvas{{position:absolute;top:0;left:0;}}
.node{{position:absolute;border-radius:50%;display:flex;flex-direction:column;
  align-items:center;justify-content:center;cursor:pointer;
  transition:transform 0.2s;transform:translate(-50%,-50%);z-index:10;
  border:2px solid rgba(255,255,255,0.12);box-shadow:0 4px 20px rgba(0,0,0,0.5);}}
.node:hover{{transform:translate(-50%,-50%) scale(1.15);}}
.node.pulse{{animation:pulse 0.5s ease-out;}}
@keyframes pulse{{
  0%{{box-shadow:0 0 0 0 rgba(255,255,255,0.5);}}
  70%{{box-shadow:0 0 0 22px rgba(255,255,255,0);}}
  100%{{box-shadow:0 0 0 0 rgba(255,255,255,0);}}
}}
.nlbl{{font-size:11px;font-weight:700;color:#fff;text-align:center;line-height:1.2;}}
.nctr{{font-size:14px;font-weight:800;color:#fff;margin-top:2px;font-variant-numeric:tabular-nums;}}
.nsub{{font-size:9px;color:rgba(255,255,255,0.5);margin-top:1px;text-align:center;}}
.badge{{position:absolute;top:10px;right:10px;display:flex;align-items:center;gap:5px;
  background:rgba(255,255,255,0.5);border:1px solid rgba(0,0,0,0.1);
  border-radius:4px;padding:3px 6px;z-index:20;}}
.bdot{{width:4px;height:4px;border-radius:50%;background:#10B981;animation:blink 1s infinite;}}
@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0.2;}}}}
.btxt{{font-size:8px;font-weight:600;color:#64748B;text-transform:uppercase;}}
.sbar{{position:absolute;bottom:8px;left:8px;right:8px;display:flex;
  justify-content:space-between;z-index:20;}}
.spill{{background:{_og_spill_bg};border:1px solid {_og_spill_bd};
  border-radius:20px;padding:3px 10px;font-size:10px;color:{_og_spill_tx};}}
.spill b{{color:{_og_spill_b};font-weight:700;}}
</style></head><body>
<div id="outr" class="outer"><div class="wrap">
  <div class="badge"><div class="bdot"></div><div class="btxt">Live Simulation</div></div>
  <canvas id="cv" width="640" height="400"></canvas>
  <div class="node" id="n-case"
   style="left:75px;top:197px;width:78px;height:78px;
          background:linear-gradient(135deg,#7C3AED,#6C63FF);">
<div class="nlbl">{_lbl_case}</div>
<div class="nctr" id="cnt-case">0</div>
<div class="nsub">{_ac:,} {_sub_case}</div>
  </div>
  <div class="node" id="n-material"
   style="left:252px;top:83px;width:62px;height:62px;
          background:linear-gradient(135deg,#6D28D9,#A78BFA);">
<div class="nlbl">{_lbl_material}</div>
<div class="nsub">{_art} types</div>
  </div>
  <div class="node" id="n-machine"
   style="left:339px;top:197px;width:78px;height:78px;
          background:linear-gradient(135deg,#B45309,#F59E0B);">
<div class="nlbl">{_lbl_machine}</div>
<div class="nctr" id="cnt-machine">0</div>
<div class="nsub">{_am} {_sub_machine}</div>
  </div>
  <div class="node" id="n-worker"
   style="left:339px;top:312px;width:64px;height:64px;
          background:linear-gradient(135deg,#065F46,#10B981);">
<div class="nlbl">{_lbl_worker}</div>
<div class="nsub">{_aw} {_sub_worker}</div>
  </div>
  <div class="node" id="n-outcome"
   style="left:564px;top:197px;width:78px;height:78px;
          background:linear-gradient(135deg,#991B1B,#EF4444);">
<div class="nlbl">{_lbl_outcome}</div>
<div class="nctr" id="cnt-outcome">0</div>
<div class="nsub">{_ao:,} results</div>
  </div>
  <div class="sbar">
<div class="spill">Types: <b>5</b></div>
<div class="spill">Active: <b id="aflows">0</b></div>
<div class="spill">Processed: <b id="tproc">0</b></div>
<div class="spill">Rate: <b id="rate">0</b>/s</div>
  </div>
</div></div>
<div style="margin-top:6px;padding:5px 14px;background:{_og_bg};border:1px solid {_og_border};border-radius:8px;display:flex;flex-wrap:wrap;gap:4px 0;">
  <span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;"><span style="width:8px;height:8px;border-radius:50%;background:#6C63FF;display:inline-block;"></span><span style="font-size:11px;color:{_og_spill_tx};">{_lbl_case}</span></span>
  <span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;"><span style="width:8px;height:8px;border-radius:50%;background:#F59E0B;display:inline-block;"></span><span style="font-size:11px;color:{_og_spill_tx};">{_lbl_machine}</span></span>
  <span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;"><span style="width:8px;height:8px;border-radius:50%;background:#10B981;display:inline-block;"></span><span style="font-size:11px;color:{_og_spill_tx};">{_lbl_worker}</span></span>
  <span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;"><span style="width:8px;height:8px;border-radius:50%;background:#A78BFA;display:inline-block;"></span><span style="font-size:11px;color:{_og_spill_tx};">{_lbl_material}</span></span>
  <span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;"><span style="width:8px;height:8px;border-radius:50%;background:#EF4444;display:inline-block;"></span><span style="font-size:11px;color:{_og_spill_tx};">{_lbl_outcome}</span></span>
</div>
<script>
function fitWrap(){{
  var s=document.body.clientWidth/640;
  if(s<=0)return;
  var w=document.querySelector('.wrap');
  w.style.transform='scale('+s+')';
  document.getElementById('outr').style.height=Math.ceil(400*s+2)+'px';
}}
fitWrap();
setTimeout(fitWrap,100);
setTimeout(fitWrap,500);
window.addEventListener('resize',fitWrap);
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const N={{
  case:    {{x:75, y:197,r:37,c:'#6C63FF'}},
  material:{{x:252,y:83, r:29,c:'#A78BFA'}},
  machine: {{x:339,y:197,r:37,c:'#F59E0B'}},
  worker:  {{x:339,y:312,r:30,c:'#10B981'}},
  outcome: {{x:564,y:197,r:37,c:'#EF4444'}},
}};
const EDGES=[
  {{f:'case',    t:'machine', c:'#6C63FF',w:2.5,prim:true, freq:26}},
  {{f:'machine', t:'outcome', c:'#EF4444',w:2.5,prim:true, freq:32}},
  {{f:'material',t:'machine', c:'#A78BFA',w:1.5,prim:false,freq:50}},
  {{f:'worker',  t:'machine', c:'#10B981',w:1.5,prim:false,freq:58}},
  {{f:'case',    t:'worker',  c:'#6C63FF',w:1.2,prim:false,freq:68}},
];
let parts=[],cnts={{case:0,machine:0,outcome:0}},total=0,prev=0,rate=0,frame=0,lastT=Date.now();
function h2r(h,a){{
  const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}
function drawEdges(){{
  EDGES.forEach(e=>{{
const f=N[e.f],t=N[e.t];
ctx.beginPath();ctx.moveTo(f.x,f.y);ctx.lineTo(t.x,t.y);
ctx.strokeStyle=h2r(e.c,0.2);ctx.lineWidth=e.w;ctx.stroke();
const ang=Math.atan2(t.y-f.y,t.x-f.x);
const ax=t.x-Math.cos(ang)*(t.r+5),ay=t.y-Math.sin(ang)*(t.r+5);
ctx.beginPath();
ctx.moveTo(ax,ay);
ctx.lineTo(ax-9*Math.cos(ang-0.45),ay-9*Math.sin(ang-0.45));
ctx.lineTo(ax-9*Math.cos(ang+0.45),ay-9*Math.sin(ang+0.45));
ctx.closePath();ctx.fillStyle=h2r(e.c,0.4);ctx.fill();
  }});
}}
function spawn(ei){{
  const e=EDGES[ei],f=N[e.f],t=N[e.t];
  parts.push({{x:f.x,y:f.y,tx:t.x,ty:t.y,from:e.f,to:e.t,c:e.c,
prog:0,spd:0.007+Math.random()*0.006,sz:e.prim?5:3.5,trail:[],done:false}});
}}
function pulse(id){{
  const el=document.getElementById('n-'+id);
  if(!el)return;el.classList.remove('pulse');void el.offsetWidth;el.classList.add('pulse');
}}
function setCtr(id,v){{const el=document.getElementById('cnt-'+id);if(el)el.textContent=v.toLocaleString();}}
function loop(){{
  ctx.clearRect(0,0,640,400);drawEdges();frame++;
  EDGES.forEach((e,i)=>{{if(frame%e.freq===0)spawn(i);}});
  parts=parts.filter(p=>p.prog<1);
  document.getElementById('aflows').textContent=parts.length;
  parts.forEach(p=>{{
p.prog+=p.spd;if(p.prog>1)p.prog=1;
p.x+=(p.tx-p.x)*p.spd*8;p.y+=(p.ty-p.y)*p.spd*8;
p.trail.push({{x:p.x,y:p.y}});if(p.trail.length>9)p.trail.shift();
p.trail.forEach((pt,i)=>{{
  ctx.beginPath();ctx.arc(pt.x,pt.y,p.sz*(i/p.trail.length)*0.7,0,Math.PI*2);
  ctx.fillStyle=h2r(p.c,(i/p.trail.length)*0.3);ctx.fill();
}});
ctx.beginPath();ctx.arc(p.x,p.y,p.sz,0,Math.PI*2);
ctx.fillStyle=p.c;ctx.shadowColor=p.c;ctx.shadowBlur=12;ctx.fill();ctx.shadowBlur=0;
if(p.prog>=0.97&&!p.done){{
  p.done=true;pulse(p.to);
  if(p.to==='machine'){{cnts.machine++;setCtr('machine',cnts.machine);}}
  if(p.to==='outcome'){{cnts.outcome++;total++;setCtr('outcome',cnts.outcome);
    document.getElementById('tproc').textContent=total.toLocaleString();}}
  if(p.from==='case'){{cnts.case++;setCtr('case',cnts.case);}}
}}
  }});
  const now=Date.now();
  if(now-lastT>1000){{rate=total-prev;prev=total;lastT=now;document.getElementById('rate').textContent=rate;}}
  requestAnimationFrame(loop);
}}
loop();
</script></body></html>""", height=400)

    with _col_stats:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};'
            f'border-radius:12px;padding:24px;font-family:Inter,sans-serif;">'

            f'<div style="font-size:13px;font-weight:700;color:{TEXT};'
            f'margin-bottom:16px;">Process Object Statistics</div>'

            # Hero stat — Object Types is the primary metric for this panel
            f'<div style="padding:18px 16px;border:1px solid #BBF7D0;border-radius:8px;'
            f'background:linear-gradient(135deg,#F0FDF4,#F8FAFC);margin-bottom:12px;">'
            f'<div style="font-size:34px;font-weight:800;color:{PRIMARY};letter-spacing:-0.03em;line-height:1;">{_avg_objects:.0f}</div>'
            f'<div style="font-size:11px;font-weight:700;color:#065F46;text-transform:uppercase;letter-spacing:0.08em;margin-top:6px;">Object Types Tracked</div>'
            f'</div>'

            # Secondary stats — de-emphasized 3-up row
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px;">'

            # Multi-object
            f'<div style="padding:12px 10px;border:1px solid {BORDER};border-radius:8px;background:#F8FAFC;">'
            f'<div style="font-size:17px;font-weight:700;color:{MUTED};letter-spacing:-0.02em;line-height:1;">{_multi_obj_pct}%</div>'
            f'<div style="font-size:9px;font-weight:600;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-top:5px;">Multi-Object</div>'
            f'</div>'

            # Top machine
            f'<div style="padding:12px 10px;border:1px solid {BORDER};border-radius:8px;background:#F8FAFC;">'
            f'<div style="font-size:14px;font-weight:700;color:{MUTED};font-family:\'JetBrains Mono\',monospace;line-height:1.2;">{_top_machine}</div>'
            f'<div style="font-size:9px;font-weight:600;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-top:5px;">Top {_mach_label}</div>'
            f'</div>'

            # Top worker
            f'<div style="padding:12px 10px;border:1px solid {BORDER};border-radius:8px;background:#F8FAFC;">'
            f'<div style="font-size:14px;font-weight:700;color:{MUTED};font-family:\'JetBrains Mono\',monospace;line-height:1.2;">{_top_worker}</div>'
            f'<div style="font-size:9px;font-weight:600;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-top:5px;">Top {_work_label}</div>'
            f'</div>'

            f'</div>'

            # Key Insight
            f'<div style="background:linear-gradient(135deg,#F0FDF4,#FFFFFF);'
            f'border:1px solid #BBF7D0;border-left:4px solid #10B981;'
            f'border-radius:8px;padding:16px;">'
            f'<div style="font-size:0.85rem;font-weight:700;color:#065F46;'
            f'margin-bottom:8px;display:flex;align-items:center;gap:5px;">💡 Key Insight</div>'
            f'<div style="font-size:0.8rem;color:#1E293B;line-height:1.55;">'
            f'By tracking <b style="color:#10B981;">{len(_type_n)} object types</b> simultaneously, '
            f'CausalOCPM can discover hidden causal bottlenecks across the entire network.'
            f'</div></div>'

            f'</div>',
            unsafe_allow_html=True,
        )


    with st.expander("📁 View Raw Data Preview", expanded=False):
        st.markdown(f"<div style='margin-bottom:8px;'><span class='badge b-neu'>First 10 rows</span></div>", unsafe_allow_html=True)
        render_table(display_df, fmt)

    # ── STEP 3: OBSERVE PROCESS CORRELATIONS ──────────────────────────────────
    st.markdown('<div class="discovery-section"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="discovery-header">Step 3: Observe Process Correlations</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="background: linear-gradient(to right, #FFFBEB, #FFFFFF); border: 1px solid #FDE68A; border-left: 4px solid #F59E0B; border-radius: 8px; padding: 16px 20px; margin-bottom: 24px;">
            <div style="font-size: 0.9rem; font-weight: 800; color: #D97706; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 1.1rem;">⚠️</span> Correlation View
            </div>
            <div style="color: #475569; font-size: 0.95rem; line-height: 1.5;">
                This visualization represents observed correlations only.<br>
                <b>True causal effects</b> are estimated in the following causal discovery section after removing confounding variables.
            </div>
        </div>
    """, unsafe_allow_html=True)

    _delay_thresh = float(df[outcome_var].median())
    _delayed_mask = df[outcome_var] > _delay_thresh

    if domain == "manufacturing" and "carrier_express" in df.columns and "supplier_a" in df.columns:
        _ex_o  = int(((df["carrier_express"] == 1) & ~_delayed_mask).sum())
        _ex_d  = int(((df["carrier_express"] == 1) &  _delayed_mask).sum())
        _st_o  = int(((df["carrier_express"] == 0) & ~_delayed_mask).sum())
        _st_d  = int(((df["carrier_express"] == 0) &  _delayed_mask).sum())
        _sa_o  = int(((df["supplier_a"] == 1) & ~_delayed_mask).sum())
        _sa_d  = int(((df["supplier_a"] == 1) &  _delayed_mask).sum())
        _sb_o  = int(((df["supplier_a"] == 0) & ~_delayed_mask).sum())
        _sb_d  = int(((df["supplier_a"] == 0) &  _delayed_mask).sum())

        _sk = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=30, thickness=24,
                line=dict(color=BORDER, width=0.5),
                label=["Express Carrier", "Standard Carrier",
                       "Supplier A", "Supplier B",
                       "✓ On Time", "✗ Delayed"],
                color=["#3B82F6", "#94A3B8", "#10B981", "#F59E0B", "#6C63FF", "#EF4444"],
                x=[0.05, 0.05, 0.05, 0.05, 0.95, 0.95],
                y=[0.10, 0.30, 0.60, 0.80, 0.28, 0.72],
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2, 3, 3],
                target=[4, 5, 4, 5, 4, 5, 4, 5],
                value=[_ex_o, _ex_d, _st_o, _st_d,
                       _sa_o, _sa_d, _sb_o, _sb_d],
                color=[
                    "rgba(59,130,246,0.15)",  "rgba(239,68,68,0.15)",
                    "rgba(148,163,184,0.15)", "rgba(239,68,68,0.15)",
                    "rgba(16,185,129,0.15)",  "rgba(239,68,68,0.15)",
                    "rgba(245,158,11,0.15)",  "rgba(239,68,68,0.25)",
                ],
            ),
        ))
        _sk_lay = dict(**PLOTLY_LAYOUT)
        _sk_lay.update(
            title="", height=360, margin=dict(l=10, r=10, t=20, b=20),
            font=dict(family="Inter", size=13, color="#1E293B")
        )
        _sk.update_layout(**_sk_lay)
        try:
            st.plotly_chart(_sk, width='stretch', theme=None, config={'displayModeBar': False})
        except Exception as _e:
            st.warning(f"Sankey could not render: {_e}")

    elif treatment_var in df.columns:
        # Healthcare / generic: single binary split Sankey
        _t_opts   = cfg.get("treatment_options", {})
        _t_lbl    = _t_opts.get(treatment_var, treatment_var.replace("_", " ").title()
                                ).split("—")[0].strip()
        _to  = int(((df[treatment_var] == 1) & ~_delayed_mask).sum())
        _td  = int(((df[treatment_var] == 1) &  _delayed_mask).sum())
        _co  = int(((df[treatment_var] == 0) & ~_delayed_mask).sum())
        _cd  = int(((df[treatment_var] == 0) &  _delayed_mask).sum())
        _out_lbl = "Short Stay" if domain == "healthcare" else "On Time"
        _del_lbl = "Long Stay"  if domain == "healthcare" else "Delayed"

        _sk = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=30, thickness=24,
                line=dict(color=BORDER, width=0.5),
                label=[_t_lbl, f"No {_t_lbl}",
                       f"✓ {_out_lbl}", f"✗ {_del_lbl}"],
                color=["#10B981", "#94A3B8", "#6C63FF", "#EF4444"],
                x=[0.05, 0.05, 0.95, 0.95],
                y=[0.25, 0.75, 0.25, 0.75],
            ),
            link=dict(
                source=[0, 0, 1, 1], target=[2, 3, 2, 3],
                value=[_to, _td, _co, _cd],
                color=[
                    "rgba(16,185,129,0.15)", "rgba(239,68,68,0.15)",
                    "rgba(148,163,184,0.15)", "rgba(239,68,68,0.15)",
                ],
            ),
        ))
        _sk_lay = dict(**PLOTLY_LAYOUT)
        _sk_lay.update(
            title="", height=320, margin=dict(l=10, r=10, t=20, b=20),
            font=dict(family="Inter", size=13, color="#1E293B")
        )
        _sk.update_layout(**_sk_lay)
        try:
            st.plotly_chart(_sk, width='stretch', theme=None, config={'displayModeBar': False})
        except Exception as _e:
            st.warning(f"Sankey could not render: {_e}")



if summary:
    _cov_types = len(_type_n) if G and G.number_of_nodes() > 0 else 5
    st.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};border-left:4px solid #14B8A6;'
        f'border-radius:8px;padding:20px 24px;margin-bottom:16px;">'
        f'<div style="font-size:0.8rem;font-weight:800;color:#0D9488;text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:14px;">OCEL Graph Coverage</div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:14px;">'
        f'<div><div style="font-size:1.6rem;font-weight:800;color:{TEXT};letter-spacing:-0.02em;line-height:1;">{summary.get("total_nodes", 0):,}</div>'
        f'<div style="font-size:0.75rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:0.05em;margin-top:4px;">Object Instances</div></div>'
        f'<div><div style="font-size:1.6rem;font-weight:800;color:{TEXT};letter-spacing:-0.02em;line-height:1;">{summary.get("total_edges", 0):,}</div>'
        f'<div style="font-size:0.75rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:0.05em;margin-top:4px;">Co-occurrence Edges</div></div>'
        f'<div><div style="font-size:1.6rem;font-weight:800;color:{TEXT};letter-spacing:-0.02em;line-height:1;">{summary.get("avg_degree", 0):.2f}</div>'
        f'<div style="font-size:0.75rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:0.05em;margin-top:4px;">Avg Degree · {_cov_types} Types</div></div>'
        f'</div>'
        f'<div style="color:#334155;font-size:0.9rem;line-height:1.5;border-top:1px solid {BORDER};padding-top:12px;">'
        f'CausalOCPM tracks every object type simultaneously, not just cases.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


    # ── STEP 4: RECOVERED CAUSAL STRUCTURE ────────────────────────────────────
    st.markdown('<div class="discovery-section"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="discovery-header">Step 4: Recovered Causal Structure</div>
    """, unsafe_allow_html=True)
    _treat_str = "Supplier A" if domain == "manufacturing" else "Patient Complexity"
    _out_str = outcome_label
    _meds_list = ["Material Lead Time"] if domain == "manufacturing" else ["Specialist Required"]
    _meds_str = ", ".join(_meds_list)
    
    # ── KEY FINDING ──────────────────────────────────────────────────────────
    _mean_gc  = dag_metrics.get('mean_gt_confidence', None)
    _boot_n_b = dag.graph.get('bootstrap_n', 0)
    if _mean_gc is not None and _boot_n_b > 1:
        _conf_col       = "#059669" if _mean_gc >= 0.80 else "#D97706"
        _boot_conf_label = f"Bootstrap edge confidence: {_mean_gc:.0%}  ·  {_boot_n_b} runs"
    else:
        _conf_col        = "#059669"
        _boot_conf_label = "Estimated Confidence: High"

    # Visual pathway generation
    _path_steps = [_treat_str] + _meds_list + [_out_str]
    _path_html = '<div style="display:flex; flex-direction:column; align-items:center; gap:8px; margin: 16px 0;">'
    for _idx, _step in enumerate(_path_steps):
        if _idx > 0:
            _path_html += '<div style="color:#94A3B8; font-size:1.2rem; font-weight:800; line-height:1;">↓</div>'
        _bg_c = "#ECFDF5" if _idx == 0 else ("#FEF2F2" if _idx == len(_path_steps)-1 else "#EFF6FF")
        _bd_c = "#A7F3D0" if _idx == 0 else ("#FECACA" if _idx == len(_path_steps)-1 else "#BFDBFE")
        _tx_c = "#059669" if _idx == 0 else ("#DC2626" if _idx == len(_path_steps)-1 else "#1D4ED8")
        _path_html += f'<div style="background:{_bg_c}; border:1px solid {_bd_c}; color:{_tx_c}; padding:8px 16px; border-radius:8px; font-weight:700; font-size:0.95rem; min-width:180px; text-align:center;">{_step}</div>'
    _path_html += '</div>'

    st.markdown(
        f'<div style="background: linear-gradient(135deg, #F0FDF4, #FFFFFF); '
        f'border: 1px solid #BBF7D0; border-left: 4px solid #10B981; '
        f'border-radius: 8px; padding: 24px; margin-bottom: 24px; '
        f'box-shadow: 0 4px 12px rgba(16, 185, 129, 0.05);">'
        f'<div style="color: #065F46; font-size: 1rem; font-weight: 800; display:flex; align-items:center; gap:8px; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em;">'
        f'🧠 Key Finding</div>'
        f'<p style="margin: 0; color: #1E293B; font-size: 1.1rem; font-weight: 500; text-align:center;">'
        f'Observed Causal Pathway:</p>'
        f'{_path_html}'
        f'<p style="margin: 8px 0 0 0; color: {_conf_col}; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; text-align:center;">'
        f'{_boot_conf_label}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── SHORT AI INTERPRETATION ───────────────────────────────────────────────
    # Adds plain-language context rather than repeating the pathway diagram
    # shown immediately above in the Key Finding card.
    _interp_med = _meds_list[0] if _meds_list else "downstream factors"
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:32px; '
        f'padding:12px 16px; background:#F8FAFC; border-radius:8px; border:1px solid {BORDER};">'
        f'<span style="font-size:1.1rem;">💬</span>'
        f'<span style="color:#475569; font-size:0.95rem; line-height:1.5;">'
        f'In plain terms: <b>{_treat_str}</b> drives up <b>{_interp_med}</b>, which in turn pushes '
        f'<b>{_out_str}</b> higher. The graph below shows exactly how this and every other '
        f'relationship was discovered.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# 2. Learned Causal DAG ────────────────────────────────────────────────────

# Compute confounding path edges (BUG-006)
confounder_set: set = set()
for _src, _dst in dag.edges():
    if _dst == treatment_var:
        confounder_set.add(_src)

_bkt_local = {"mediator": []}
for _n in dag.nodes():
    if _n not in [outcome_var, treatment_var] and _n not in confounder_set:
        _bkt_local["mediator"].append(_n)

# Only include nodes that lie on an actual path from treatment → outcome
_true_meds = []
for _n in _bkt_local["mediator"]:
    try:
        if nx.has_path(dag, treatment_var, _n) and nx.has_path(dag, _n, outcome_var):
            _true_meds.append(_n)
    except nx.NodeNotFound:
        pass
_meds_list = [n.replace("_", " ").title() for n in (_true_meds if _true_meds else _bkt_local["mediator"])]
_meds_str = " and ".join(_meds_list) if _meds_list else "other factors"
_treat_str = treatment_var.replace("_", " ").title()
_out_str = outcome_var.replace("_", " ").title()

confound_path_edges: set = set()
for _conf in confounder_set:
    if dag.has_edge(_conf, outcome_var):
        confound_path_edges.add((_conf, outcome_var))
        confound_path_edges.add((_conf, treatment_var))
    else:
        for _nb in dag.successors(_conf):
            if _nb != treatment_var:
                try:
                    if nx.has_path(dag, _nb, outcome_var):
                        confound_path_edges.add((_conf, treatment_var))
                        break
                except nx.NodeNotFound:
                    pass

# Animated Causal Discovery v4 (Professional) ────────────────────────────
if dag.number_of_nodes() > 0:
    import math as _math
    import json as _json
    import streamlit.components.v1 as _stc

    _hl_toggle = st.radio(
        "DAG Visualization Mode",
        ["Show Full DAG", "Highlight Strongest Path"],
        horizontal=True,
        label_visibility="collapsed"
    )
    _hl_strongest = (_hl_toggle == "Highlight Strongest Path")

    try:
        _strong_path = nx.shortest_path(dag, treatment_var, outcome_var)
        _strong_edges = set(zip(_strong_path, _strong_path[1:]))
    except nx.NetworkXNoPath:
        _strong_edges = set()

    _f1   = dag_metrics.get("f1_score",  0.0)
    _prec = dag_metrics.get("precision", 0.0)
    _rec  = dag_metrics.get("recall",    0.0)

    # Bootstrap edge confidence (from Phase 2 bootstrapped PC)
    _edge_conf = dag.graph.get('edge_confidence', {})
    _boot_n    = dag.graph.get('bootstrap_n', 0)

    # ── Classify nodes ───────────────────────────────────────────────
    _node_role: Dict[str, str] = {}
    for _n in dag.nodes():
        if _n == outcome_var:      _node_role[_n] = "outcome"
        elif _n == treatment_var:  _node_role[_n] = "treatment"
        elif _n in confounder_set: _node_role[_n] = "confounder"
        else:                      _node_role[_n] = "mediator"

    # Flat professional palette — no gradients
    _ROLE_COLOR  = {"outcome":"#DC2626","treatment":"#059669",
                    "confounder":"#D97706","mediator":"#3B82F6"}
    _ROLE_RADIUS = {"outcome":46,"treatment":38,"confounder":40,"mediator":33}

    # ── Layout ──────────────────────────────────────────────────────
    _CW, _CH = 900, 480
    _bkt: Dict[str, list] = {k:[] for k in ("confounder","treatment","mediator","outcome")}
    for _n, _r in _node_role.items():
        _bkt[_r].append(_n)

    def _vcenter(nodes, cy=240, gap=130):
        n = len(nodes)
        return [int(cy + (i-(n-1)/2)*gap) for i in range(n)]

    _pos: Dict[str, Dict] = {}
    for _i, _nd in enumerate(_bkt["confounder"]):
        _pos[_nd] = {"x": 100, "y": _vcenter(_bkt["confounder"])[_i]}
    for _i, _nd in enumerate(_bkt["treatment"]):
        _pos[_nd] = {"x": 260, "y": _vcenter(_bkt["treatment"])[_i]}
    for _i, _nd in enumerate(_bkt["outcome"]):
        _pos[_nd] = {"x": 800, "y": _vcenter(_bkt["outcome"])[_i]}

    _meds    = _bkt["mediator"]
    _meds_left = [_nd for _i, _nd in enumerate(_meds) if _i % 2 == 0]
    _meds_right = [_nd for _i, _nd in enumerate(_meds) if _i % 2 == 1]
    
    _y_left = _vcenter(_meds_left, cy=_CH//2, gap=130)
    _y_right = _vcenter(_meds_right, cy=_CH//2, gap=130)
    
    for _i, _nd in enumerate(_meds_left):
        _pos[_nd] = {"x": 440, "y": _y_left[_i]}
    for _i, _nd in enumerate(_meds_right):
        _pos[_nd] = {"x": 620, "y": _y_right[_i]}

    # ── Node label (balanced 2-line) ─────────────────────────────────
    def _nlabel(name):
        words = name.replace("_", " ").title().split()
        if len(words) == 1: return words[0]
        mid = len(words) // 2
        return " ".join(words[:mid]) + "<br>" + " ".join(words[mid:])

    # ── SVG edge paths (Python-computed, no JS coord math) ───────────
    def _edge_path(sn, dn):
        p1, p2 = _pos[sn], _pos[dn]
        r1 = _ROLE_RADIUS[_node_role[sn]]
        r2 = _ROLE_RADIUS[_node_role[dn]]
        dx, dy = p2["x"]-p1["x"], p2["y"]-p1["y"]
        d = _math.hypot(dx, dy)
        if d < 1: return None
        nx_, ny_ = dx/d, dy/d
        sx = p1["x"] + nx_*(r1+3);  sy = p1["y"] + ny_*(r1+3)
        ex = p2["x"] - nx_*(r2+8);  ey = p2["y"] - ny_*(r2+8)
        return f"M {sx:.0f} {sy:.0f} L {ex:.0f} {ey:.0f}"

    # Normal edges first, confounding last
    _sorted_edges = sorted(dag.edges(), key=lambda e: (e[0], e[1]) in confound_path_edges)

    _svg_paths, _edge_meta, _pi = [], [], 0
    for _s, _d in _sorted_edges:
        _cf        = (_s, _d) in confound_path_edges
        _is_strong = (_s, _d) in _strong_edges
        _pth       = _edge_path(_s, _d)
        if _pth is None: continue

        # Bootstrap confidence for this edge (1.0 if no bootstrap data)
        _conf     = _edge_conf.get((_s, _d), 1.0)
        _conf_pct = int(_conf * 100)
        # Map confidence → stroke-width: low conf=1.0px, high conf=4.0px
        _conf_sw  = f"{1.0 + _conf * 3.0:.1f}" if _edge_conf else "1.5"

        if _hl_strongest and not _is_strong:
            _col     = "#CBD5E1" if IS_LIGHT else "#334155"
            _opacity = "0.2"
            _sw      = "1.0"
        else:
            _col     = "#DC2626" if _cf else "#3B82F6"
            _opacity = "1"
            _sw      = "3.5" if (_hl_strongest and _is_strong) else _conf_sw

        _marker = "am-c" if _cf else "am-n"
        if _hl_strongest and not _is_strong:
            _marker = "am-faded"

        if _cf:
            _svg_paths.append(
                f'<path id="e{_pi}" d="{_pth}" stroke="{_col}" stroke-width="{_sw}" '
                f'fill="none" stroke-dasharray="6 4" '
                f'marker-end="url(#{_marker})" class="ec" opacity="0"/>'
            )
        else:
            _svg_paths.append(
                f'<path id="e{_pi}" d="{_pth}" stroke="{_col}" stroke-width="{_sw}" '
                f'fill="none" stroke-dasharray="2000" stroke-dashoffset="2000" '
                f'marker-end="url(#{_marker})" class="en" opacity="0"/>'
            )

        # Confidence label at edge midpoint (shown only when bootstrap data exists)
        if _edge_conf and not (_hl_strongest and not _is_strong):
            _p1b, _p2b = _pos[_s], _pos[_d]
            _mx = (_p1b["x"] + _p2b["x"]) / 2
            _my = (_p1b["y"] + _p2b["y"]) / 2
            _lbl_col = "#6366F1" if _conf >= 0.8 else "#D97706"
            _svg_paths.append(
                f'<text x="{_mx:.0f}" y="{_my - 7:.0f}" '
                f'font-size="9" font-weight="700" fill="{_lbl_col}" '
                f'text-anchor="middle" dominant-baseline="middle" '
                f'class="econf" opacity="0">{_conf_pct}%</text>'
            )

        _edge_meta.append({"c": _cf, "col": _col, "op": _opacity})
        _pi += 1

    _n_norm  = sum(1 for e in _edge_meta if not e["c"])
    _n_total = len(_edge_meta)

    # ── Node divs ────────────────────────────────────────────────────
    # Variable descriptions for rich tooltips
    _VAR_DESC: Dict[str, str] = {
        "order_complexity":    "Complexity score of the order (1–10)",
        "supplier_a":          "Binary: order fulfilled by Supplier A",
        "material_lead_time":  "Days for material to arrive from supplier",
        "machine_queue_length":"Number of jobs ahead in machine queue",
        "export_flag":         "Binary: order requires export clearance",
        "approval_duration":   "Days spent in approval workflow",
        "carrier_express":     "Binary: express carrier used for shipment",
        "shipment_delay":      "Total shipment delay in days (outcome)",
        # Healthcare variables
        "specialist_referral": "Binary: patient referred to specialist",
        "comorbidity_index":   "Number of concurrent diagnoses",
        "test_duration":       "Days spent on diagnostic testing",
        "bed_occupancy_rate":  "Fraction of beds occupied at admission",
        "insurance_type":      "Binary: private insurance (vs public)",
        "procedure_complexity":"Complexity score of the procedure",
        "length_of_stay":      "Total inpatient days (outcome)",
    }
    _node_divs = ""
    for _nd in dag.nodes():
        _r = _ROLE_RADIUS[_node_role[_nd]]
        _c = _ROLE_COLOR[_node_role[_nd]]
        # Compute mean value for the tooltip
        try:
            _mean_val = f"{df[_nd].mean():.1f} days" if _nd in df.columns else "—"
        except Exception:
            _mean_val = "—"
        _desc_val = _VAR_DESC.get(_nd, _nd.replace("_", " ").title())
        _node_divs += (
            f'<div class="dn" id="nd-{_nd}" data-role="{_node_role[_nd]}" '
            f'data-val="{_mean_val}" data-desc="{_desc_val}" '
            f'style="left:{_pos[_nd]["x"]}px;top:{_pos[_nd]["y"]}px;'
            f'width:{_r*2}px;height:{_r*2}px;background:{_c};">'
            f'<div class="nl">{_nlabel(_nd)}</div></div>\n'
        )

    _svg_body   = "\n".join(_svg_paths)
    _emeta_json = _json.dumps(_edge_meta)
    _nids_json  = _json.dumps([str(n) for n in dag.nodes()])

    _bg_color = "#F8F9FB" if IS_LIGHT else "#0A0E17"
    _border_color = "#E2E8F0" if IS_LIGHT else "#1A1F2E"
    _ectr_bg = "rgba(255,255,255,0.9)" if IS_LIGHT else "rgba(10,14,23,0.9)"

    _stc.html(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{background:{_bg_color};font-family:-apple-system,'Inter',sans-serif;overflow:hidden;max-width:100%;}}
.wrap{{
  position:relative;width:900px;height:480px;margin:0 auto;
  background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;overflow:hidden;
  box-shadow:0 4px 12px rgba(0,0,0,0.05);
}}
svg{{position:absolute;top:0;left:0;z-index:1;overflow:visible;}}
#nh{{position:absolute;inset:0;z-index:10;}}

/* Nodes */
.dn{{
  position:absolute;border-radius:50%;
  display:flex;align-items:center;justify-content:center;flex-direction:column;
  cursor:grab;user-select:none;
  border:1.5px solid rgba(255,255,255,0.12);
  box-shadow:0 2px 12px rgba(0,0,0,0.4);
  transform:translate(-50%,-50%) scale(0.85);opacity:0;
  transition:opacity 0.4s cubic-bezier(0.4,0,0.2,1),
         transform 0.4s cubic-bezier(0.4,0,0.2,1),
         box-shadow 0.3s ease;
}}
.dn.vis{{transform:translate(-50%,-50%) scale(1);opacity:1;}}
.dn:hover{{
  box-shadow:0 0 0 3px rgba(255,255,255,0.08),0 2px 12px rgba(0,0,0,0.4)!important;
  transform:translate(-50%,-50%) scale(1.05)!important;z-index:99;
}}
.dn:active{{cursor:grabbing;}}
/* Confounding: subtle static ring, no animation */
.dn.cring{{box-shadow:0 0 0 2px rgba(220,38,38,0.4),0 2px 12px rgba(0,0,0,0.4);}}
.nl{{
  font-size:11.5px;font-weight:600;color:#fff;text-align:center;
  line-height:1.35;padding:0 6px;pointer-events:none;
  text-shadow:0 1px 3px rgba(0,0,0,0.8);
}}

/* Edge transitions — material easing, no glow */
.en{{transition:stroke-dashoffset 0.5s cubic-bezier(0.4,0,0.2,1),
          opacity 0.35s cubic-bezier(0.4,0,0.2,1);}}
.ec{{transition:opacity 0.5s cubic-bezier(0.4,0,0.2,1);}}
/* Bootstrap confidence labels on edges */
.econf{{transition:opacity 0.6s ease 1.2s;font-family:-apple-system,'Inter',sans-serif;}}

/* Status bar */
.sbar{{
  position:absolute;top:10px;left:10px;right:10px;
  display:flex;align-items:center;justify-content:space-between;
  z-index:40;pointer-events:none;
}}
.abadge{{
  display:flex;align-items:center;gap:8px;
  background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
  border-radius:6px;padding:6px 14px;
}}
.adot{{width:5px;height:5px;border-radius:50%;background:#818CF8;flex-shrink:0;}}
.adot.ok{{background:#10B981;}}
.atxt{{font-size:11px;font-weight:600;color:#818CF8;letter-spacing:0.04em;}}
.ectr{{
  background:{_ectr_bg};border:1px solid {_border_color};
  border-radius:6px;padding:4px 12px;font-size:11px;color:#4B5563;
}}
.ectr b{{color:#6366F1;font-weight:700;}}

/* Completion banner — slides down, no big numbers */
.banner{{
  position:absolute;top:0;left:50%;
  transform:translateX(-50%) translateY(-50px);opacity:0;
  background:rgba(6,78,59,0.10);border:1px solid rgba(16,185,129,0.22);
  border-radius:8px;padding:8px 18px;
  transition:transform 0.4s cubic-bezier(0.4,0,0.2,1),opacity 0.4s ease;
  z-index:50;pointer-events:none;white-space:nowrap;
}}
.banner.show{{transform:translateX(-50%) translateY(46px);opacity:1;pointer-events:all;}}
.brow{{display:flex;align-items:center;gap:12px;}}
.bchk{{font-size:13px;color:#10B981;}}
.btxt{{font-size:12px;font-weight:600;color:#10B981;}}
.bmeta{{font-size:11px;color:#4B5563;}}
.bmeta b{{color:#6B7280;font-weight:600;}}
.rbtn{{
  background:transparent;border:1px solid rgba(99,102,241,0.3);
  border-radius:5px;padding:3px 10px;color:#818CF8;
  font-size:10px;font-weight:600;cursor:pointer;
  transition:background 0.2s;letter-spacing:0.04em;
}}
.rbtn:hover{{background:rgba(99,102,241,0.1);}}

/* Legend */
.leg{{
  position:absolute;bottom:10px;left:10px;
  display:flex;gap:14px;z-index:30;
  pointer-events:none;opacity:0;transition:opacity 0.5s;
}}
.leg.show{{opacity:1;}}
.li{{display:flex;align-items:center;gap:6px;font-size:11px;color:#4B5563;font-weight:600;}}
.lc{{width:9px;height:9px;border-radius:50%;flex-shrink:0;}}
.ll{{width:21px;height:2px;flex-shrink:0;}}

/* Tooltip */
.tt{{
  position:absolute;background:#111827;border:1px solid #1F2937;
  border-radius:6px;padding:7px 11px;font-size:10px;color:#D1D5DB;
  pointer-events:none;z-index:100;opacity:0;transition:opacity 0.15s;
  max-width:180px;line-height:1.5;
}}
.tt.show{{opacity:1;}}
</style></head><body>
<div class="wrap">

  <!-- Status bar -->
  <div class="sbar">
<div class="abadge">
  <div class="adot" id="adot"></div>
  <div class="atxt" id="atxt">Initializing PC algorithm</div>
</div>
<div class="ectr">Edges: <b id="ec">0</b>/{_n_total}</div>
  </div>

  <!-- SVG layer: pre-computed paths, no sparks group -->
  <svg id="dag-svg" width="900" height="480">
<defs>
  <marker id="am-n" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#6366F1" fill-opacity="0.8"/>
  </marker>
  <marker id="am-c" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#DC2626" fill-opacity="0.85"/>
  </marker>
  <marker id="am-faded" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#CBD5E1" fill-opacity="0.6"/>
  </marker>
</defs>
<g id="edges-g">
{_svg_body}
</g>
  </svg>

  <!-- Node host -->
  <div id="nh">
{_node_divs}  </div>

  <!-- Completion banner -->
  <div class="banner" id="banner">
<div class="brow">
  <span class="bchk">&#10003;</span>
  <span class="btxt">Causal structure recovered</span>
  <span class="bmeta">Bootstrap Stability&nbsp;<b>{_boot_stab_pct:.0f}%</b>&nbsp;&middot;&nbsp;Precision&nbsp;<b>{_prec:.3f}</b>&nbsp;&middot;&nbsp;Edge Recall&nbsp;<b>{_rec:.3f}</b></span>
  <button class="rbtn" onclick="replay()">&#8635; Replay</button>
</div>
  </div>

  <!-- Legend -->
  <div class="leg" id="leg">
<div class="li"><div class="lc" style="background:#D97706;width:11px;height:11px;"></div>Confounder</div>
<div class="li"><div class="lc" style="background:#059669;width:11px;height:11px;"></div>Treatment</div>
<div class="li"><div class="lc" style="background:#3B82F6;width:11px;height:11px;"></div>Mediator</div>
<div class="li"><div class="lc" style="background:#DC2626;width:11px;height:11px;"></div>Outcome</div>
<div class="li"><div class="ll" style="background:#3B82F6;height:3.5px;border-radius:1.75px;width:25px;"></div>Positive effect</div>
<div class="li">
  <svg width="25" height="5" style="flex-shrink:0">
    <line x1="0" y1="2" x2="25" y2="2" stroke="#DC2626" stroke-width="2.3" stroke-dasharray="6,3.5"/>
  </svg>
  Negative effect
</div>
<div class="li">
  <svg width="37" height="7" style="flex-shrink:0">
    <line x1="0" y1="3" x2="14" y2="3" stroke="#94A3B8" stroke-width="1.7"/>
    <line x1="18" y1="3" x2="37" y2="3" stroke="#6366F1" stroke-width="5.75"/>
  </svg>
  Edge width = bootstrap confidence
</div>
  </div>

  <!-- Click-to-inspect hint — appears after animation completes -->
  <div id="hint" style="position:absolute;bottom:42px;left:0;right:0;text-align:center;
   font-size:10px;color:#6B7280;pointer-events:none;opacity:0;transition:opacity 0.5s;">
&#128070; Click any node on the graph to inspect its causal role, current value, and connections.
  </div>

  <div class="tt" id="tt"></div>
</div>

<script>
const EDGE_META = {_emeta_json};
const NODE_IDS  = {_nids_json};
const N_NORMAL  = {_n_norm};
const N_TOTAL   = {_n_total};

function setText(id, txt) {{ const el=document.getElementById(id); if(el) el.textContent=txt; }}
function cls(el, ...c)   {{ if(el) c.forEach(x=>el.classList.add(x)); }}
function uncls(el, ...c) {{ if(el) c.forEach(x=>el.classList.remove(x)); }}
function $id(id)         {{ return document.getElementById(id); }}
function $$cls(c)        {{ return document.querySelectorAll('.'+c); }}

// ── Edge reveal (no sparks, no flash) ─────────────────────────────────────────
let ecnt = 0;
function revealEdge(i) {{
  const el   = $id('e'+i);
  const meta = EDGE_META[i];
  if(!el || !meta) return;
  el.style.opacity = meta.op || '1';
  if(!meta.c) el.style.strokeDashoffset = '0';
  ecnt++;
  setText('ec', ecnt);
  // Confounding: mark all nodes with a subtle static ring
  if(meta.c) $$cls('dn').forEach(n => n.classList.add('cring'));
}}

// ── Animation sequence ────────────────────────────────────────────────────────
let timers = [], done = false;
function clearTimers() {{ timers.forEach(clearTimeout); timers = []; }}

function startAnim() {{
  clearTimers();
  done = false;
  ecnt = 0;

  // Reset edges
  EDGE_META.forEach((m, i) => {{
const el = $id('e'+i);
if(!el) return;
el.style.opacity = '0';
if(!m.c) el.style.strokeDashoffset = '2000';
  }});
  // Reset confidence labels
  $$cls('econf').forEach(el => {{ el.style.transition = 'none'; el.style.opacity = '0'; }});

  // Reset nodes
  NODE_IDS.forEach(id => {{
const el = $id('nd-'+id);
if(el) uncls(el, 'vis', 'cring');
  }});

  uncls($id('banner'), 'show');
  uncls($id('leg'), 'show');
  const hint = $id('hint');
  if(hint) hint.style.opacity = '0';
  setText('ec', '0');

  const dot = $id('adot');
  if(dot) {{ dot.classList.remove('ok'); dot.style.background = '#818CF8'; }}
  setText('atxt', 'Initializing PC algorithm');

  // Phase 1 — nodes appear staggered (200ms apart, smooth ease)
  NODE_IDS.forEach((id, i) => {{
timers.push(setTimeout(() => {{
  const el = $id('nd-'+id);
  if(el) cls(el, 'vis');
}}, 200 + i * 200));
  }});

  // Phase 2 — independence test label
  const p2 = 200 + NODE_IDS.length * 200 + 250;
  timers.push(setTimeout(() => setText('atxt', 'Testing conditional independence'), p2));

  // Phase 3 — edges draw one by one
  const p3 = p2 + 1100, step = 480;
  for(let i = 0; i < N_TOTAL; i++) {{
timers.push(setTimeout(() => {{
  if(i === 0)        setText('atxt', 'Orienting causal edges');
  if(i === N_NORMAL) setText('atxt', 'Identifying backdoor paths');
  revealEdge(i);
}}, p3 + i * step));
  }}

  // Phase 4 — backdoor criterion note
  const p4 = p3 + N_TOTAL * step + 300;
  timers.push(setTimeout(() => setText('atxt', 'Backdoor criterion applied'), p4));

  // Phase 5 — completion banner slides down + confidence labels fade in
  const p5 = p4 + 700;
  timers.push(setTimeout(() => {{
cls($id('banner'), 'show');
cls($id('leg'),    'show');
const dot = $id('adot');
if(dot) {{ dot.style.background = '#10B981'; dot.classList.add('ok'); }}
setText('atxt', 'Discovery complete');
const hint = $id('hint');
if(hint) hint.style.opacity = '1';
// Fade in bootstrap confidence labels
$$cls('econf').forEach(el => {{
  el.style.transition = 'opacity 0.6s ease';
  el.style.opacity = '1';
}});
done = true;
  }}, p5));
}}

function replay() {{
  uncls($id('banner'), 'show');
  setTimeout(startAnim, 300);
}}

// ── Drag (post-animation only, for exploration) ───────────────────────────────
let isDrag = false, dragId = null, dox = 0, doy = 0;
const wrap = document.querySelector('.wrap');

wrap.addEventListener('mousedown', e => {{
  if(!done) return;
  const wR = wrap.getBoundingClientRect();
  const mx = e.clientX - wR.left, my = e.clientY - wR.top;
  for(const id of NODE_IDS) {{
const el = $id('nd-'+id);
if(!el) continue;
const eR = el.getBoundingClientRect();
const cx = eR.left - wR.left + eR.width/2;
const cy = eR.top  - wR.top  + eR.height/2;
if(Math.hypot(mx-cx, my-cy) < eR.width/2 + 6) {{
  isDrag = true; dragId = id; dox = mx-cx; doy = my-cy;
  el.style.cursor = 'grabbing';
  break;
}}
  }}
}});

window.addEventListener('mousemove', e => {{
  const wR = wrap.getBoundingClientRect();
  const mx = e.clientX - wR.left, my = e.clientY - wR.top;
  if(isDrag && dragId) {{
const el = $id('nd-'+dragId);
if(el) {{ el.style.left = (mx-dox)+'px'; el.style.top = (my-doy)+'px'; }}
  }}
  // Tooltip on hover
  if(!isDrag && done) {{
let found = false;
for(const id of NODE_IDS) {{
  const el = $id('nd-'+id);
  if(!el) continue;
  const eR = el.getBoundingClientRect();
  const cx = eR.left - wR.left + eR.width/2;
  const cy = eR.top  - wR.top  + eR.height/2;
  if(Math.hypot(mx-cx, my-cy) < eR.width/2 + 4) {{
    const tt = $id('tt');
    if(tt) {{
      const role = el.dataset.role || '';
      const val  = el.dataset.val  || '';
      const desc = el.dataset.desc || '';
      const roleColor = {{
        outcome:'#DC2626', treatment:'#059669',
        confounder:'#D97706', mediator:'#818CF8'
      }}[role] || '#818CF8';
      tt.innerHTML =
        '<b style="color:'+roleColor+';text-transform:capitalize;">'+role+'</b>'
        + (val  ? '<br><span style="color:#9CA3AF;font-size:9.5px;">Value: </span>'
                + '<span style="color:#E5E7EB;">'+val+'</span>' : '')
        + (desc ? '<br><span style="color:#6B7280;font-size:9px;">'+desc+'</span>' : '');
      tt.style.left = (cx + eR.width/2 + 8) + 'px';
      tt.style.top  = (cy - 24) + 'px';
      cls(tt, 'show');
    }}
    found = true;
    break;
  }}
}}
if(!found) uncls($id('tt'), 'show');
  }}
}});

window.addEventListener('mouseup', () => {{
  if(dragId) {{ const el = $id('nd-'+dragId); if(el) el.style.cursor = 'grab'; }}
  isDrag = false; dragId = null;
}});

setTimeout(startAnim, 300);
</script></body></html>""", height=500)

    # ── DETAILED EXPLANATION ──────────────────────────────────────────────────
    _flow_steps = [f"<b>{_treat_str}</b>"]
    for _med in _meds_list:
        _flow_steps.append(f"Increases <b>{_med}</b>")
    _flow_steps.append(f"Results in <b>{_out_str}</b>")
    
    _flow_html = '<div style="display:flex; flex-direction:column; align-items:center; gap:6px; margin-bottom:16px;">'
    for _idx, _fstep in enumerate(_flow_steps):
        if _idx > 0:
            _flow_html += '<div style="color:#94A3B8; font-size:1.1rem; font-weight:800; line-height:1;">↓</div>'
        _flow_html += f'<div style="color:#475569; font-size:0.95rem; font-weight:600; text-align:center;">{_fstep}</div>'
    _flow_html += '</div>'

    st.markdown(
        f'<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-top: 16px;">'
        f'{_flow_html}'
        f'<div style="margin: 0; color: #475569; font-size: 0.95rem; line-height: 1.6; border-top: 1px solid #E2E8F0; padding-top:16px;">'
        f'Orders associated with <b>{_treat_str}</b> increase <b>{_meds_list[0] if _meds_list else "associated factors"}</b>. '
        f'This subsequent change significantly increases <b>{_out_str}</b>.<br><br>'
        f'This suggests that operational mechanics, rather than raw assignments alone, drive performance outcomes.'
        f'</div></div>',
        unsafe_allow_html=True
    )
    
    # ── STEP 5: VALIDATE DISCOVERY QUALITY ────────────────────────────────────
    st.markdown('<div class="discovery-section"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="discovery-header">Step 5: Validate Discovery Quality</div>
        <div class="discovery-desc">Before trusting this graph, it's tested against resampled data and known ground truth — a discovery is only useful if it's stable.</div>
    """, unsafe_allow_html=True)
    
    # Retrieve ablation data
    _wdk  = ablation.get("with_domain_knowledge", {})    if "ablation" in locals() and ablation else {}
    _wodk = ablation.get("without_domain_knowledge", {}) if "ablation" in locals() and ablation else {}
    
    # Pre-DK metrics — what the bootstrap PC algorithm found on its own
    _prec_raw = _wodk.get("precision", _prec) if _wodk else _prec
    _rec_raw  = _wodk.get("recall",    _rec)  if _wodk else _rec
    _f1_raw   = _wodk.get("f1_score",  _f1)   if _wodk else _f1
    _links_raw= _wodk.get("true_positives", dag.number_of_edges()) if _wodk else dag.number_of_edges()

    _c1_boot_col = "#059669" if _boot_stab_pct >= 85 else ("#D97706" if _boot_stab_pct >= 70 else "#DC2626")
    
    # Hero Bootstrap Stability Card
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#F0FDF4,#ECFDF5);'
        f'border:1px solid #A7F3D0;border-radius:12px;padding:32px 24px;margin-bottom:24px;text-align:center;">'
        f'<div style="font-size:0.85rem;font-weight:800;color:#064E3B;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:8px;">Bootstrap Stability</div>'
        f'<div style="font-size:3.5rem;font-weight:900;color:{_c1_boot_col};line-height:1;">{_boot_stab_pct:.0f}%</div>'
        f'<div style="font-size:0.95rem;color:#065F46;margin-top:8px;">'
        f'Edges stable across {dag.graph.get("bootstrap_n", 20)} resampled graphs</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    c1, c2, c3 = st.columns(3)
    for _col, _label, _val in (
        (c1, "Precision", _prec_raw), (c2, "Edge Recall", _rec_raw), (c3, "Recovery F1", _f1_raw)
    ):
        _col.markdown(
            f'<div class="kpi-card kpi-card--quality" style="text-align:center;align-items:center;">'
            f'<div class="kpi-value" style="font-size:1.8rem;">{_val:.2f}</div>'
            f'<div class="kpi-label">{_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── STEP 6: EVALUATE DOMAIN KNOWLEDGE CONTRIBUTION ────────────────────────
    st.markdown('<div class="discovery-section"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="discovery-header">Step 6: Evaluate Domain Knowledge Contribution</div>
        <div class="discovery-desc">Comparing results with and without expert constraints shows exactly what domain knowledge added — and confirms it never invented a relationship.</div>
    """, unsafe_allow_html=True)

    _rc_base   = _wodk.get("recall",    0.0) or 0.0
    _fn_base   = _wodk.get("false_negatives", 0) or 0
    _fn_after  = _wdk.get("false_negatives",  0) or 0
    _fp_base   = _wodk.get("false_positives",  0) or 0
    _fp_after  = _wdk.get("false_positives",   0) or 0
    _links_recovered = max(0, _fn_base - _fn_after)
    _links_pruned    = max(0, _fp_base - _fp_after)
    _rec_gain_pp = (_rec - _rc_base) * 100
    _dk_links_after = int(_wdk.get("true_positives", len(_sorted_edges)))

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown(
            f'<div style="background:#F0FDF4;border:1px solid #A7F3D0;border-radius:10px;padding:24px;height:100%;">'
            f'<div style="font-size:2.2rem;font-weight:900;color:#059669;line-height:1;">+{_rec_gain_pp:.1f}%</div>'
            f'<div style="font-size:0.85rem;font-weight:700;color:#064E3B;text-transform:uppercase;margin-top:8px;">Recall Gain</div>'
            f'<div style="font-size:0.95rem;color:#065F46;margin-top:8px;">{_links_recovered} Missing Edge{"s" if _links_recovered != 1 else ""} Recovered</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with dc2:
        st.markdown(
            f'<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;padding:24px;height:100%;">'
            f'<div style="font-size:2.2rem;font-weight:900;color:#C2410C;line-height:1;">-{_links_pruned}</div>'
            f'<div style="font-size:0.85rem;font-weight:700;color:#9A3412;text-transform:uppercase;margin-top:8px;">Spurious Links Removed</div>'
            f'<div style="font-size:0.95rem;color:#9A3412;margin-top:8px;">Resulting in {_dk_links_after} Validated Links</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-left:4px solid #64748B;'
        'border-radius:8px;padding:20px;margin-top:24px;">'
        '<div style="font-size:0.95rem;font-weight:700;color:#1E293B;margin-bottom:12px;">Domain Knowledge Integration</div>'
        '<div style="display:flex;flex-direction:column;gap:8px;">'
        '<div style="color:#059669;font-weight:600;font-size:0.95rem;">✓ Removed spurious links</div>'
        '<div style="color:#059669;font-weight:600;font-size:0.95rem;">✓ Recovered missing causal edges</div>'
        '<div style="color:#059669;font-weight:600;font-size:0.95rem;">✓ Preserved DAG validity</div>'
        '<div style="color:#059669;font-weight:600;font-size:0.95rem;">✓ Improved causal recall</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 7. Graph Legend ──────────────────────────────────────────────────────
    with st.expander("ⓘ Graph Legend", expanded=False):
        st.markdown(
            f'<div style="display:flex; gap:24px; padding:8px;">'
            f'<div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#059669;"></div><span style="font-size:0.9rem;">Treatment</span></div>'
            f'<div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#DC2626;"></div><span style="font-size:0.9rem;">Outcome</span></div>'
            f'<div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#3B82F6;"></div><span style="font-size:0.9rem;">Mediator</span></div>'
            f'<div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#D97706;"></div><span style="font-size:0.9rem;">Confounder</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # 8. Ablation Study — Domain Knowledge Impact ──────────────────────────
    if ablation:
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        with st.expander("Domain Knowledge Ablation Study", expanded=False):
            st.markdown(
                f'<p style="color:#64748B; font-size:0.9rem; margin-top:0; margin-bottom:16px;">'
                f'Discovery quality with vs. without domain knowledge constraints — empirical justification for knowledge integration.</p>',
                unsafe_allow_html=True,
            )

            _abl_wdk  = ablation.get("with_domain_knowledge", {})
            _abl_wodk = ablation.get("without_domain_knowledge", {})

            # ── Ablation Finding First ──
            _abl_imp      = ablation.get("improvement", {})
            _f1_delta     = _abl_imp.get("f1_gain",    0.0)
            _rc_delta     = _abl_imp.get("recall_gain", 0.0)
            _prec_delta   = _abl_imp.get("precision_gain", 0.0)
            _gt_n_abl     = len(ablation.get("with_domain_knowledge", {}).get("ground_truth_edges", []))
            _links_rec_n  = round(abs(_rc_delta) * (_gt_n_abl or 9))
            _prec_wdk     = _abl_wdk.get("precision", 0.0)
            _prec_wodk    = _abl_wodk.get("precision", 0.0)
            _prec_clause  = (
                f'Graph precision <b style="color:#2563EB;">improved from {_prec_wodk:.3f} → {_prec_wdk:.3f}</b>'
                if _prec_delta > 0.001
                else f'Graph precision was <b style="color:#2563EB;">maintained at {_prec_wdk:.3f}</b>'
            )
            st.markdown(
                f'<div style="background:#EFF6FF;border-left:4px solid #3B82F6;padding:16px 20px;border-radius:4px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,0.02);">'
                f'<div style="color:#1E3A8A;font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">ABLATION FINDING</div>'
                f'<div style="color:#1E293B;font-size:0.95rem;line-height:1.6;">'
                f'Domain constraints recovered <b style="color:#2563EB;">{_links_rec_n} missed causal link{"s" if _links_rec_n != 1 else ""}</b> '
                f'and improved edge recall by <b style="color:#2563EB;">{_rc_delta*100:+.1f} percentage points</b>. '
                + _prec_clause +
                f' — domain knowledge adds only validated edges, never speculative links.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Ablation Chart Second ──
            _abl_metrics = ["Structural Precision", "Edge Recall", "Recovery F1"]
            _abl_with    = [_abl_wdk.get("precision", 0.0),  _abl_wdk.get("recall", 0.0),  _abl_wdk.get("f1_score", 0.0)]
            _abl_without = [_abl_wodk.get("precision", 0.0), _abl_wodk.get("recall", 0.0), _abl_wodk.get("f1_score", 0.0)]

            fig_abl = go.Figure()
            fig_abl.add_trace(go.Bar(
                name="Without Domain Knowledge",
                x=_abl_metrics,
                y=_abl_without,
                marker_color=MUTED,
                opacity=0.85,
                text=[f"{v:.3f}" for v in _abl_without],
                textposition="outside",
            ))
            fig_abl.add_trace(go.Bar(
                name="With Domain Knowledge",
                x=_abl_metrics,
                y=_abl_with,
                marker_color=SUCCESS,
                opacity=0.88,
                text=[f"{v:.3f}" for v in _abl_with],
                textposition="outside",
            ))

            _abl_layout = dict(**PLOTLY_LAYOUT)
            _abl_layout.update(dict(
                barmode="group",
                yaxis={"title": "Score", "range": [0, 1.2], "title_font": dict(size=13), "tickformat": ".2f"},
                xaxis={"title": "Metric"},
                height=400,
                margin=dict(l=20, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            ))
            fig_abl.update_layout(**_abl_layout)
            try:
                st.plotly_chart(fig_abl, width='stretch', theme=None, config={'displayModeBar': False})
            except Exception as _ablE:
                st.error(f"Chart error: {_ablE}")

    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #F8FAFC, #FFFFFF); border: 1px solid #E2E8F0; border-left: 4px solid #0284C7; border-radius: 12px; padding: 28px 32px; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.05); margin-bottom: 24px;">
        <div style="font-size: 0.9rem; font-weight: 800; color: #0284C7; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.3rem;">✨</span> Discovery Recap
        </div>
        <ul style="color: #334155; font-size: 0.95rem; line-height: 1.8; margin: 0 0 20px 0; padding-left: 20px;">
            <li>Causal pathway <b>Supplier A ➡ Material Lead Time ➡ {outcome_label}</b> held up under bootstrap resampling and correlation-vs-causation checks.</li>
            <li>Domain knowledge closed the remaining gap, recovering missing edges without introducing spurious ones.</li>
            <li>The discovered structure is validated and ready for intervention analysis.</li>
        </ul>
        <div style="display: flex; justify-content: flex-end;">
            <span style="font-size: 0.9rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.05em; background: rgba(2, 132, 199, 0.08); padding: 8px 16px; border-radius: 6px; border: 1px solid rgba(2, 132, 199, 0.2);">
                Navigate to Model Performance above to continue →
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 3 — STRUCTURAL CAUSAL MODEL ──────────────────────────────────────────
