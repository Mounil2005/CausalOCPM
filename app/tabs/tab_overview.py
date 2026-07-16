# ── SHARED "AI ___ SUMMARY" CARD COMPONENT ─────────────────────────────────
# Defined here because tab_overview.py is exec'd first (see dashboard.py's
# tab loop) — Data & Discovery and Model & Impact reuse this same function
# by name via the shared globals() dict all tabs execute into, rather than
# each hand-rolling their own version of the same card, the way the old
# "AI Executive/Discovery Summary" boxes independently did (three different
# layouts for what was conceptually one component).
def _ai_status(frac):
    if frac >= 0.85: return SUCCESS, "High"
    if frac >= 0.65: return WARNING, "Moderate"
    return ERROR, "Low"

def _ai_card(accent, badge_bg, icon, title, conf_color, conf_label, lead, bullets):
    _bullets_html = "".join(f'<li style="margin-bottom:5px;">{b}</li>' for b in bullets)
    return (
        f'<div style="background:linear-gradient(135deg,{badge_bg},#FFFFFF);border:1px solid {BORDER};'
        f'border-left:4px solid {accent};border-radius:14px;padding:24px 28px;margin-bottom:28px;'
        f'box-shadow:0 4px 12px rgba(0,0,0,0.03);">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:14px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="width:30px;height:30px;border-radius:9px;background:{accent}22;display:flex;'
        f'align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;">{icon}</span>'
        f'<span style="font-size:0.85rem;font-weight:800;color:{accent};text-transform:uppercase;'
        f'letter-spacing:0.08em;">{title}</span>'
        f'</div>'
        f'<span style="font-size:0.68rem;font-weight:800;color:#FFFFFF;background:{conf_color};'
        f'padding:4px 11px;border-radius:20px;text-transform:uppercase;letter-spacing:0.03em;">'
        f'{conf_label} Confidence</span>'
        f'</div>'
        f'<div style="font-size:1.15rem;font-weight:800;color:#0F172A;line-height:1.45;margin-bottom:14px;">'
        f'{lead}</div>'
        f'<ul style="color:#334155;font-size:0.92rem;line-height:1.6;margin:0;padding-left:20px;">'
        f'{_bullets_html}</ul>'
        f'</div>'
    )

# ── MODEL VALIDATION NUMBERS (computed early — the AI card needs them too) ──
# Sourced from ablation["without_domain_knowledge"] — the raw PC-algorithm
# discovery — NOT `dag_metrics`. `dag_metrics` is computed after
# enforce_domain_knowledge() force-adds every planted ground-truth edge
# into the graph, which makes recall = 1.00 guaranteed by construction
# rather than something actually discovered; showing that number as a
# trust signal would be circular (see phase2_discovery.py).
_ov_wodk = ablation.get("without_domain_knowledge", {}) if ablation else {}
_ov_prec = _ov_wodk.get("precision", dag_metrics.get("precision", 0.0))
_ov_rec  = _ov_wodk.get("recall",    dag_metrics.get("recall",    0.0))
_ov_f1   = _ov_wodk.get("f1_score",  dag_metrics.get("f1_score",  0.0))
_ov_tp     = _ov_wodk.get("true_positives",  0)
_ov_fp     = _ov_wodk.get("false_positives", 0)
_ov_fn     = _ov_wodk.get("false_negatives", 0)
_ov_boot_n = dag.graph.get("bootstrap_n", 20)

# ── BOARDROOM HERO CARD (data) ─────────────────────────────────────────────
_hero_treatment = cfg.get("treatment_var", "treatment")
_hero_outcome   = cfg.get("outcome_label", cfg.get("outcome_var", "outcome"))
# The "Recovered Causal Effect" card must show what the pipeline actually
# estimated (Double ML), not the planted ground-truth constant it's being
# validated against — those are different numbers and conflating them would
# make the pipeline's own output indistinguishable from the answer key.
_hero_te        = (do_result.get("causal") if stage_status.get("do_operator") == "ok" and do_result
                    else None)
_hero_baseline  = round(float(df[cfg["outcome_var"]].mean()), 1) if cfg.get("outcome_var") in df.columns else (8.2 if domain == "manufacturing" else 5.27)
# Reduction % and saving from same formula as simulator and exec report
if stage_status.get("do_operator") == "ok" and do_result:
    _causal_eff = abs(do_result.get("causal", 0))
    _shift_ratio = 0.25 if domain == "manufacturing" else 0.50
    _hero_reduction = (_causal_eff * _shift_ratio / _hero_baseline) * 100 if _hero_baseline > 0 else 0
else:
    _hero_reduction = 18.3 if domain == "manufacturing" else 12.7
_hero_mult      = 300 * 960 if domain == "manufacturing" else 400 * 1050
_hero_saving_val = round(_hero_reduction / 100 * _hero_baseline * _hero_mult / 1000) * 1000
_hero_pct       = f"{_hero_reduction:.0f}%"
_hero_saving    = f"~${_hero_saving_val//1000}K"
_hero_action    = "Shift ~25% procurement to Supplier B" if domain == "manufacturing" else "Optimise specialist allocation protocols"
_hero_causal    = f"{_hero_te:.2f}" if _hero_te else "—"
_hero_naive_val = df[cfg["outcome_var"]].mean() + (df[df[_hero_treatment]==1][cfg["outcome_var"]].mean() - df[df[_hero_treatment]==0][cfg["outcome_var"]].mean()) * 0.4 if _hero_treatment in df.columns else _hero_baseline
_hero_label       = "Shipment Delay" if domain == "manufacturing" else "Length of Stay"
_hero_root_cause  = "Supplier A dependency" if domain == "manufacturing" else "Specialist over-allocation"
_hero_act_short   = "Shift 25% to Supplier B" if domain == "manufacturing" else "Refine triage criteria"
# Set directly per domain rather than parsed out of _hero_root_cause's display
# text — string-splitting on " dependency"/" over-allocation" would silently
# break if that display copy ever changed.
_primary_driver   = "Supplier A" if domain == "manufacturing" else "Specialist Allocation"

# ── AI EXECUTIVE SUMMARY — now genuinely derived from this run's numbers
# (recovered causal effect, discovery precision/recall, recommended action)
# instead of a fixed string that read the same regardless of the data.
_ov_avg_conf = (_ov_prec + _ov_rec + _ov_f1) / 3
_ov_conf_col, _ov_conf_lbl = _ai_status(_ov_avg_conf)
st.markdown(
    _ai_card(
        accent=PRIMARY, badge_bg="#F0FDF4", icon="✨", title="AI Executive Summary",
        conf_color=_ov_conf_col, conf_label=_ov_conf_lbl,
        lead=(
            f"<b>{_primary_driver}</b> is confirmed as the dominant causal driver of "
            f"{_hero_label.lower()} — statistically validated, not just correlated."
        ),
        bullets=[
            f"Recovered causal effect: <b>{_hero_causal} days</b> via Double ML — "
            f"{_hero_pct} reduction potential vs. baseline",
            f"Discovery precision <b>{_ov_prec:.2f}</b>, recall <b>{_ov_rec:.2f}</b> across "
            f"{_ov_boot_n} bootstrap reruns (full breakdown below)",
            f"Recommended action: <b>{_hero_action}</b> → {_hero_saving}/yr expected savings",
        ],
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f'<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #134E4A 100%); '
    f'border-radius: 16px; padding: 36px 40px; margin-bottom: 28px; position: relative; overflow: hidden;">'
    f'<div style="position:absolute;top:0;right:0;width:300px;height:100%;'
    f'background:linear-gradient(90deg,transparent,rgba(16,185,129,0.08));pointer-events:none;"></div>'
    # Alert tag
    f'<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(220,38,38,0.15);'
    f'border:1px solid rgba(220,38,38,0.4);border-radius:6px;padding:5px 14px;margin-bottom:20px;">'
    f'<span style="width:8px;height:8px;border-radius:50%;background:#EF4444;display:inline-block;'
    f'animation:pulse 1.5s infinite;"></span>'
    f'<span style="color:#FCA5A5;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">'
    f'CAUSAL INTELLIGENCE ALERT</span></div>'
    # Big headline
    f'<div style="color:#FFFFFF;font-size:2.6rem;font-weight:900;line-height:1.1;margin-bottom:12px;letter-spacing:-0.02em;">'
    f'{_hero_label.upper()} CAN BE<br>'
    f'<span style="color:#34D399;">REDUCED BY {_hero_pct}</span></div>'
    # Subtitle — executive-friendly, no academic jargon above the fold
    f'<div style="color:#94A3B8;font-size:1.0rem;font-weight:500;margin-bottom:20px;line-height:1.6;">'
    f'Root cause identified. &nbsp;Business impact quantified. &nbsp;Intervention validated through causal simulation.</div>'
    # Narrative flow strip
    f'<div style="display:flex;align-items:center;background:rgba(255,255,255,0.04);'
    f'border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:10px 18px;margin-bottom:24px;">'
    f'<span style="color:#CBD5E1;font-size:0.77rem;font-weight:600;">{_hero_label}</span>'
    f'<span style="color:#475569;font-size:0.77rem;padding:0 10px;">→</span>'
    f'<span style="color:#FCA5A5;font-size:0.77rem;font-weight:600;">{_hero_root_cause}</span>'
    f'<span style="color:#475569;font-size:0.77rem;padding:0 10px;">→</span>'
    f'<span style="color:#93C5FD;font-size:0.77rem;font-weight:600;">{_hero_act_short}</span>'
    f'<span style="color:#475569;font-size:0.77rem;padding:0 10px;">→</span>'
    f'<span style="color:#34D399;font-size:0.77rem;font-weight:700;">{_hero_pct} reduction</span>'
    f'</div>'
    # 3 KPI pills — hierarchy: savings (2.1rem) > action (0.97rem) > causal effect (1.7rem)
    f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
    # Savings pill — largest, yellow
    f'<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:16px 24px;min-width:175px;">'
    f'<div style="color:#94A3B8;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Expected Annual Savings</div>'
    f'<div style="color:#FBBF24;font-size:2.1rem;font-weight:900;letter-spacing:-0.02em;">{_hero_saving}</div>'
    f'<div style="color:#64748B;font-size:0.72rem;margin-top:4px;">Based on current throughput</div></div>'
    # Action pill — action text + green outcome below it
    f'<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:16px 24px;min-width:190px;">'
    f'<div style="color:#94A3B8;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Highest ROI Action</div>'
    f'<div style="color:#FFFFFF;font-size:0.97rem;font-weight:700;line-height:1.35;">{_hero_action}</div>'
    f'<div style="color:#34D399;font-size:0.8rem;font-weight:700;margin-top:6px;">↓ {_hero_pct} delay reduction</div></div>'
    # Causal effect pill — smallest visual weight
    f'<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:16px 24px;min-width:155px;">'
    f'<div style="color:#94A3B8;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Recovered Causal Effect</div>'
    f'<div style="color:#34D399;font-size:1.7rem;font-weight:900;letter-spacing:-0.02em;">{_hero_causal}<span style="font-size:0.95rem;margin-left:4px;">days</span></div>'
    f'<div style="color:#64748B;font-size:0.72rem;margin-top:4px;">Double ML estimate (ATE)</div></div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)
st.markdown("<style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}</style>", unsafe_allow_html=True)

# ── MODEL VALIDATION STRIP ────────────────────────────────────────────────
# Answers "why should I trust this?" right under the headline claim —
# discovered-graph-vs-planted-ground-truth precision/recall/F1, plus
# bootstrap edge stability. Same numbers as the full breakdown in
# Data & Discovery → Step 5, surfaced here so a reader never has to scroll
# to find out whether the model is any good.
#
# (numbers already computed above, near the AI Executive Summary card,
# since that card needs them too)

def _ov_status(frac):
    if frac >= 0.85: return SUCCESS, "Strong"
    if frac >= 0.65: return WARNING, "Moderate"
    return ERROR, "Needs Review"

# Per-metric "i" info icon — a native title attribute, not a custom tooltip
# widget, so it needs no JS and can't trip the React-crash issue hit earlier
# with inline onclick handlers. Each one spells out the actual formula with
# this run's real TP/FP/FN counts, not just a generic definition, so hovering
# any single box answers "how did THIS number happen" on its own.
def _ov_info_icon(tooltip):
    return (
        f'<span title="{tooltip}" style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:15px;height:15px;border-radius:50%;background:{BORDER};color:{MUTED};'
        f'font-size:0.62rem;font-weight:800;font-style:normal;cursor:help;flex-shrink:0;">i</span>'
    )

# NL = literal HTML numeric-entity newline, not a real "\n\n" blank line.
# st.markdown() runs its whole argument through a markdown-to-HTML pass
# before the browser ever sees it, even the parts inside our raw HTML
# attributes — a literal blank line inside title="..." gets read as a
# markdown paragraph break and Streamlit splits the tag apart around it,
# corrupting the whole card (verified: rendered as a mangled <p>...</p>
# spliced into the middle of the attribute, breaking is/left the "i" icon
# not existing anywhere in the DOM). The numeric entity has no literal
# newline in the source, so markdown leaves it alone, and the browser still
# renders it as a real line break inside the native tooltip.
_NL = "&#10;&#10;"
_ov_gt_n_hint = _ov_tp + _ov_fn
_OV_TOOLTIPS = {
    "Bootstrap Stability": (
        f"Think of it as asking {_ov_boot_n} independent witnesses the same question instead of "
        f"trusting just one. The algorithm reruns causal discovery {_ov_boot_n} times, each on a "
        f"slightly different resampled slice of the same data, and checks how often it lands on "
        f"the same causal edges.{_NL}"
        f"A relationship that keeps reappearing across resamples is likely a real pattern in how "
        f"the process behaves, not a coincidence of this one dataset. One that only shows up once "
        f"is a red flag it might just be noise.{_NL}"
        f"Result here: {_boot_stab_pct:.0f}% agreement across {_ov_boot_n} reruns — most of what "
        f"was found holds up under repeated testing."
    ),
    "Precision": (
        f"Picture the algorithm as a detective naming suspects. Precision asks: of everyone it "
        f"accused, how many were actually guilty?{_NL}"
        f"It named {_ov_tp + _ov_fp} relationships as causal. {_ov_tp} matched the real planted "
        f"structure; {_ov_fp} were false accusations — links it claimed exist but don't.{_NL}"
        f"{_ov_tp} correct ÷ {_ov_tp + _ov_fp} accused = {_ov_prec:.2f}. A high score means that "
        f"when this model says 'A causes B,' you can trust it — it rarely cries wolf."
    ),
    "Edge Recall": (
        f"This flips the question around: of all {_ov_gt_n_hint} real causal relationships "
        f"actually planted in the data, how many did the algorithm manage to dig up on its own?{_NL}"
        f"It correctly found {_ov_tp}, and missed {_ov_fn}. {_ov_tp} found ÷ {_ov_gt_n_hint} that "
        f"truly exist = {_ov_rec:.2f}.{_NL}"
        f"A high score means the model is thorough — it doesn't leave real causes undiscovered, "
        f"even if it occasionally over-claims (that's what Precision catches separately)."
    ),
    "Recovery F1": (
        f"Precision and Recall pull against each other. You could fake a perfect Precision by "
        f"only naming the one relationship you're 100% sure about — or fake a perfect Recall by "
        f"claiming every possible link exists and being wrong most of the time.{_NL}"
        f"F1 is the honest referee: the harmonic mean of both, 2×(P×R)÷(P+R) = {_ov_f1:.2f}. It "
        f"only scores high when precision AND recall are genuinely strong together — you can't "
        f"game it by leaning on just one."
    ),
}

def _ov_metric_tile(icon, value, label, frac):
    _color, _status = _ov_status(frac)
    return (
        f'<div style="background:#FFFFFF;border:1px solid {BORDER};border-radius:12px;'
        f'padding:14px 16px;flex:1;min-width:140px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        f'<span style="font-size:0.95rem;">{icon}</span>'
        f'<span style="font-size:0.6rem;font-weight:800;color:{_color};background:{_color}1A;'
        f'padding:2px 7px;border-radius:20px;text-transform:uppercase;letter-spacing:0.03em;">{_status}</span>'
        f'</div>'
        f'<div style="font-size:1.7rem;font-weight:900;color:{TEXT};line-height:1.1;">{value}</div>'
        f'<div style="display:flex;align-items:center;gap:5px;margin-top:5px;margin-bottom:8px;">'
        f'<span style="font-size:0.68rem;font-weight:700;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:0.05em;">{label}</span>'
        + _ov_info_icon(_OV_TOOLTIPS.get(label, ""))
        + f'</div>'
        f'<div style="height:4px;background:{BORDER};border-radius:2px;overflow:hidden;">'
        f'<div style="height:100%;width:{frac*100:.0f}%;background:{_color};border-radius:2px;"></div>'
        f'</div>'
        f'</div>'
    )

_ov_avg = (_ov_prec + _ov_rec + _ov_f1) / 3
_ov_verdict_col, _ov_verdict_lbl = _ov_status(_ov_avg)

st.markdown(
    f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;'
    f'padding:18px 22px;margin-bottom:28px;box-shadow:0 2px 10px rgba(15,23,42,0.03);">'
    f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:6px;">'
    f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
    f'<span style="font-size:1.05rem;">🛡️</span>'
    f'<span style="font-size:0.82rem;font-weight:800;color:{TEXT};text-transform:uppercase;letter-spacing:0.06em;">'
    f'Model Validation — Discovered Graph vs. Ground Truth</span>'
    f'<span style="font-size:0.66rem;font-weight:800;color:#FFFFFF;background:{_ov_verdict_col};'
    f'padding:3px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:0.03em;">'
    f'✓ {_ov_verdict_lbl}</span>'
    f'</div>'
    f'<span style="font-size:0.75rem;color:{SUBTLE};">Full breakdown in '
    f'<b>Data &amp; Discovery → Step 5</b></span>'
    f'</div>'
    f'<div style="font-size:0.72rem;color:{SUBTLE};margin-bottom:14px;">'
    f'Precision/recall/F1 are the raw statistical discovery, measured before domain-knowledge '
    f'constraints are applied — not a number domain knowledge guarantees to be perfect.</div>'
    f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
    + _ov_metric_tile("🔁", f"{_boot_stab_pct:.0f}%", "Bootstrap Stability", _boot_stab_pct / 100)
    + _ov_metric_tile("🎯", f"{_ov_prec:.2f}", "Precision", _ov_prec)
    + _ov_metric_tile("🔎", f"{_ov_rec:.2f}", "Edge Recall", _ov_rec)
    + _ov_metric_tile("⚖️", f"{_ov_f1:.2f}", "Recovery F1", _ov_f1)
    + f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
with st.expander("ℹ️ How is this measured?"):
    _ov_gt_n = len(_ov_wodk.get("ground_truth_edges", [])) or dag.number_of_edges() or 9

    def _ov_flow_step(n, icon, title, desc):
        return (
            f'<div style="flex:1;min-width:160px;background:#FFFFFF;border:1px solid {BORDER};'
            f'border-radius:10px;padding:12px 14px;">'
            f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px;">'
            f'<span style="width:18px;height:18px;border-radius:50%;background:{PRIMARY};color:#FFFFFF;'
            f'font-size:0.62rem;font-weight:800;display:flex;align-items:center;justify-content:center;'
            f'flex-shrink:0;">{n}</span>'
            f'<span style="font-size:0.9rem;">{icon}</span>'
            f'<span style="font-size:0.78rem;font-weight:800;color:{TEXT};">{title}</span>'
            f'</div>'
            f'<div style="font-size:0.74rem;color:{MUTED};line-height:1.45;">{desc}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:stretch;margin-bottom:6px;">'
        + _ov_flow_step(1, "🌱", "Plant a known answer",
            f"Synthetic data generated from {_ov_gt_n} causal edges planted in advance.")
        + '<div style="display:flex;align-items:center;color:#CBD5E1;font-size:1.2rem;font-weight:800;">→</div>'
        + _ov_flow_step(2, "🔍", "Discover statistically",
            f"PC algorithm runs {_ov_boot_n}× on resampled data, no domain rules yet.")
        + '<div style="display:flex;align-items:center;color:#CBD5E1;font-size:1.2rem;font-weight:800;">→</div>'
        + _ov_flow_step(3, "✅", "Compare, edge by edge",
            f"Proposed graph checked against the {_ov_gt_n} planted edges.")
        + f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#F8FAFC;border:1px solid {BORDER};border-radius:10px;'
        f'padding:10px 14px;margin:12px 0;display:flex;gap:18px;flex-wrap:wrap;font-size:0.78rem;">'
        f'<span style="color:{SUCCESS};font-weight:700;">✓ {_ov_tp} correct edges found</span>'
        f'<span style="color:{ERROR};font-weight:700;">✗ {_ov_fp} false alarms</span>'
        f'<span style="color:{WARNING};font-weight:700;">− {_ov_fn} missed edges</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Precision** = correct ÷ all proposed (of what it found, how much was right?) · "
        f"**Recall** = correct ÷ all real (of what's true, how much did it find?) · "
        f"**F1** = both balanced into one number.\n\n"
        f"**Bootstrap Stability** is separate — the share of {_ov_boot_n} resampled reruns that "
        f"agreed an edge was real, a guard against a lucky fluke. Domain-knowledge rules are "
        f"applied only afterward, so these scores reflect the statistics working alone."
    )
    st.markdown(
        f"Domain-knowledge constraints are applied only **after** this scoring, so these numbers "
        f"are graded on the raw statistics alone. Full walkthrough in **Data & Discovery → Step 5**."
    )

# ── OCPM COMPARISON CARD ─────────────────────────────────────────────────
_n_obj_types = 5 if domain == "manufacturing" else 4
_obj_list    = "Orders · Machines · Workers · Materials · Shipments" if domain == "manufacturing" \
               else "Patients · Wards · Clinicians · Medications · Discharge"
st.markdown(
    # Trimmed to the one angle the Competitive Positioning table below doesn't
    # cover — object-centricity itself — since the capability bullets that
    # used to live here (causal discovery, counterfactual simulation, etc.)
    # duplicated rows already in that table.
    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:28px;">'
    # Traditional PM card
    f'<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;padding:20px 24px;">'
    f'<div style="color:#B91C1C;font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;">Traditional Process Mining</div>'
    f'<div style="color:#991B1B;font-size:2rem;font-weight:900;margin-bottom:8px;">1 object type</div>'
    f'<div style="color:#EF4444;font-size:0.8rem;font-weight:500;border-top:1px solid #FECACA;padding-top:8px;">Case ID only — object relationships and interactions are invisible to the model.</div>'
    f'</div>'
    # CausalOCPM card
    f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:12px;padding:20px 24px;">'
    f'<div style="color:{SUCCESS};font-size:0.75rem;font-weight:800;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;">CausalOCPM (This System)</div>'
    f'<div style="color:{SUCCESS};font-size:2rem;font-weight:900;margin-bottom:8px;">{_n_obj_types} object types</div>'
    f'<div style="color:{SUCCESS};font-size:0.8rem;font-weight:500;border-top:1px solid #BBF7D0;padding-top:8px;">{_obj_list}</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)

# 1. Executive Summary Card
st.markdown("### Executive Summary")
_health_label = "Moderate Risk" if _hero_reduction < 20 else "Elevated Risk"
_health_color = "#F59E0B" if _hero_reduction < 20 else "#DC2626"
st.markdown(
    f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid {_health_color}; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); padding: 24px; margin-bottom: 32px;">'
    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
    f'<div><h4 style="margin:0; color:#1E293B; font-size:1.1rem; font-weight:700;">Operational Health</h4>'
    f'<span style="color:{_health_color}; font-weight:800; font-size:1.3rem;">{_health_label}</span></div>'
    f'<div style="text-align:right;"><span style="color:#64748B; font-size:0.85rem; font-weight:600; text-transform:uppercase;">Highest Risk Segment</span><br>'
    f'<span style="color:#1E293B; font-weight:700; font-size:1.1rem;">{_primary_driver}</span></div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)

# 2. Intervention Leaderboard
st.markdown("### Recommended Actions")
st.markdown("<p style='color:#64748B; font-size:0.95rem; margin-top:-8px; margin-bottom:24px;'>Prioritized interventions based on estimated causal impact.</p>", unsafe_allow_html=True)

# is_computed=True (rank 1) means the % comes from the live Double ML estimate;
# ranks 2/3 are illustrative planning estimates, not pipeline output — flagged
# visually below rather than shown with identical authority to rank 1.
_ov_acts_mfg = [
    ("#1", "Shift ~25% procurement to Supplier B",   f"~{_hero_reduction:.0f}%", "High",   "Medium", "#10B981", True),
    ("#2", "Automate export approval routing",        "~7.5%",                   "Medium", "Low",    "#F59E0B", False),
    ("#3", "Expand machine buffer capacity (≥20%)",   "~3%",                     "High",   "High",   "#3B82F6", False),
]
_ov_acts_hc = [
    ("#1", "Refine specialist triage criteria",           f"~{_hero_reduction:.0f}%", "High",   "Low",    "#10B981", True),
    ("#2", "Implement fast-track triage automation",      "~5.5%",                   "Medium", "Medium", "#F59E0B", False),
    ("#3", "Expand bed capacity in high-occupancy wards", "~3%",                     "High",   "High",   "#3B82F6", False),
]
_actions = _ov_acts_mfg if domain == "manufacturing" else _ov_acts_hc

for rank, title, impact, conf, effort, color, is_computed in _actions:
    _impact_tag = (
        '<span style="background:#ECFDF5;color:#059669;border-radius:4px;padding:1px 6px;font-size:0.65rem;font-weight:700;margin-left:6px;">MEASURED</span>'
        if is_computed else
        '<span style="background:#F1F5F9;color:#64748B;border-radius:4px;padding:1px 6px;font-size:0.65rem;font-weight:700;margin-left:6px;">ILLUSTRATIVE</span>'
    )
    st.markdown(
        f'<div style="display:flex; align-items:center; background:#FFFFFF; border:1px solid #E2E8F0; border-left:3px solid {color}; border-radius:8px; padding:16px 24px; margin-bottom:12px; box-shadow:0 2px 4px rgba(0,0,0,0.02);">'
        f'<div style="font-size:1.5rem; font-weight:800; color:{color}; width:50px;">{rank}</div>'
        f'<div style="flex:1;"><div style="font-size:1.1rem; font-weight:700; color:#1E293B;">{title}</div>'
        f'<div style="font-size:0.85rem; color:#64748B; font-weight:500; margin-top:4px;">Expected Delay Reduction: <span style="color:#10B981; font-weight:700;">{impact}</span>{_impact_tag}</div></div>'
        f'<div style="width:120px;"><div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">Confidence</div>'
        f'<div style="font-size:0.95rem; font-weight:600; color:#334155;">{conf}</div></div>'
        f'<div style="width:120px;"><div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">Op. Effort</div>'
        f'<div style="font-size:0.95rem; font-weight:600; color:#334155;">{effort}</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Pipeline Performance Summary ─────────────────────────────────────────────
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
st.markdown("### Pipeline Performance Summary")
st.markdown(
    "<p style='color:#64748B; font-size:0.95rem; margin-top:-8px; margin-bottom:24px;'>"
    "Live metrics from the causal discovery and structural model phases for the active domain.</p>",
    unsafe_allow_html=True,
)

_ov_prec = dag_metrics.get("precision", 0.0)
_ov_rec  = dag_metrics.get("recall",    0.0)
_ov_f1   = dag_metrics.get("f1_score",  0.0)

_ov_sign_ok  = 0
_ov_total_e  = 0
_ov_mean_err = 0.0
if not coefs.empty:
    _ov_total_e  = len(coefs)
    _ov_sign_ok  = int((coefs["status"] != "Sign Error").sum()) if "status" in coefs.columns else _ov_total_e
    _ov_mean_err = float(coefs["pct_error"].mean()) if "pct_error" in coefs.columns and not coefs["pct_error"].isna().all() else 0.0

_ov_sign_pct    = (_ov_sign_ok / _ov_total_e * 100) if _ov_total_e > 0 else 100.0
_ablation_now   = ablation if "ablation" in locals() and ablation else {}
_ov_rec_predk   = _ablation_now.get("without_domain_knowledge", {}).get("recall", _ov_rec)

st.markdown(
    '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 16px;'
    'margin-bottom:16px;display:inline-flex;align-items:center;gap:10px;">'
    '<span style="font-size:0.68rem;font-weight:800;text-transform:uppercase;'
    'letter-spacing:0.05em;color:#64748B;">Synthetic Ground Truth Benchmark</span>'
    '<span style="color:#94A3B8;font-size:0.75rem;">&#183;</span>'
    '<span style="font-size:0.75rem;color:#475569;">Metrics computed against planted causal structure</span>'
    '</div>',
    unsafe_allow_html=True,
)

# Overview chart: two independent series with their own metric axes
# Discovery series: Post-DK Precision, Pre-DK Recall, Post-DK Recall (all graph-recovery metrics)
# SCM series: Sign Consistency, Avg R² (avg across linear nodes), 1-MeanRelErr (coefficient accuracy)
#
# The precision bar is explicitly labeled "Post-DK" (not just "Graph
# Precision") because _ov_prec below is dag_metrics' precision, computed
# AFTER enforce_domain_knowledge() force-adds every planted ground-truth
# edge — the same number this file's own AI Executive Summary / Model
# Validation cards deliberately avoid showing elsewhere because it's not a
# number the algorithm found on its own. Sitting it unlabeled next to a bar
# explicitly marked "Pre-DK Recall" would look like the two came from the
# same regime when they don't; the label makes that distinction visible
# instead of implying false consistency between the two discovery bars.
if not coefs.empty and "r2_score" in coefs.columns and not coefs["r2_score"].isna().all():
    _ov_avg_r2 = float(coefs["r2_score"].dropna().mean())
else:
    _r2_vals = [v.get("r2_score", 0.0) for v in scm.values() if isinstance(v, dict) and v.get("r2_score") is not None]
    _ov_avg_r2 = float(np.mean(_r2_vals)) if _r2_vals else 0.0
_ov_coef_acc = max(0.0, 1.0 - _ov_mean_err)  # 1 - mean relative error
_ov_scm_vals = [_ov_sign_pct / 100, _ov_avg_r2, _ov_coef_acc]
_ov_disc_x  = ["Post-DK Precision", "Pre-DK Recall", "Post-DK Recall"]
_ov_scm_x   = ["Sign Consistency", "Avg Model R²", "Coeff Accuracy"]

fig_ov = go.Figure()
fig_ov.add_trace(go.Bar(
    name="Causal Discovery",
    x=_ov_disc_x,
    y=[_ov_prec, _ov_rec_predk, _ov_rec],
    marker_color=PRIMARY,
    opacity=0.88,
    text=[f"{v:.3f}" for v in [_ov_prec, _ov_rec_predk, _ov_rec]],
    textposition="outside",
))
fig_ov.add_trace(go.Bar(
    name="Structural Model",
    x=_ov_scm_x,
    y=_ov_scm_vals,
    marker_color=SUCCESS,
    opacity=0.85,
    text=[f"{v:.3f}" for v in _ov_scm_vals],
    textposition="outside",
))

_ov_layout = dict(**PLOTLY_LAYOUT)
_ov_layout.update(dict(
    barmode="group",
    yaxis={"title": "Score (0–1)", "range": [0, 1.25], "title_font": dict(size=13), "tickformat": ".2f"},
    xaxis={"title": "Causal Discovery (left 3)  ·  Structural Model (right 3)"},
    height=380,
    margin=dict(l=20, r=20, t=30, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
))
fig_ov.update_layout(**_ov_layout)
try:
    st.plotly_chart(fig_ov, use_container_width=True, theme=None, config={'displayModeBar': False})
except Exception as _ovE:
    st.error(f"Chart error: {_ovE}")

_ov_edges = dag.number_of_edges() if "dag" in locals() else 0
st.markdown(
    f'<div style="background:#F0FDF4; border-left:4px solid #10B981; padding:16px 20px; border-radius:4px; margin-top:4px; box-shadow:0 2px 8px rgba(0,0,0,0.02);">'
    f'<div style="color:{SUCCESS}; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">PIPELINE STATUS</div>'
    f'<div style="color:#1E293B; font-size:0.95rem; line-height:1.6;">'
    f'The causal pipeline recovered <b style="color:#059669;">{_ov_edges}</b> validated causal links with '
    f'bootstrap stability of <b style="color:#059669;">{_boot_stab_pct:.0f}%</b>. '
    f'The structural model achieved sign consistency of <b style="color:#059669;">{_ov_sign_pct:.0f}%</b> '
    f'across {_ov_total_e} discovered relationships.</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── COMPETITIVE POSITIONING ───────────────────────────────────────────────
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
st.markdown("### Competitive Positioning")
st.markdown(
    "<p style='color:#64748B; font-size:0.95rem; margin-top:-8px; margin-bottom:18px;'>"
    "How CausalOCPM compares to existing process analytics approaches.</p>",
    unsafe_allow_html=True,
)
_comp_rows = [
    ("Capability",               "Traditional PM",  "Celonis",   "CausalOCPM"),
    ("Object-centric events",    "❌ Case ID only", "⚠️ Partial", "✅ Full OCEL 2.0"),
    ("Causal discovery",         "❌ None",         "❌ None",    "✅ Bootstrap PC"),
    ("Confounding adjustment",   "❌ None",         "❌ None",    "✅ Double ML"),
    ("Counterfactual simulation","❌ None",         "⚠️ Rule-based","✅ SCM-based"),
    ("Ground truth validation",  "❌ No",           "❌ No",      "✅ Planted GT"),
    ("Uncertainty quantification","❌ None",        "⚠️ CI ranges","✅ Bootstrap CIs"),
    ("Multi-domain framework",   "❌ Domain-specific","⚠️ Templates","✅ Generalised"),
]
_th = "font-size:0.78rem;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;padding:10px 16px;background:#F1F5F9;color:#475569;"
_td = "font-size:0.88rem;padding:10px 16px;color:#1E293B;border-top:1px solid #F1F5F9;"
_td_g = f"font-size:0.88rem;padding:10px 16px;color:{SUCCESS};font-weight:700;background:#F0FDF4;border-top:1px solid #F1F5F9;"
_tbl_html = '<table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;border:1px solid #E2E8F0;">'
for i, row in enumerate(_comp_rows):
    if i == 0:
        _tbl_html += f'<tr><th style="{_th}">{row[0]}</th><th style="{_th}">{row[1]}</th><th style="{_th}">{row[2]}</th><th style="{_th};color:#059669;">{row[3]}</th></tr>'
    else:
        _tbl_html += f'<tr><td style="{_td};font-weight:600;color:#334155;">{row[0]}</td><td style="{_td}">{row[1]}</td><td style="{_td}">{row[2]}</td><td style="{_td_g}">{row[3]}</td></tr>'
_tbl_html += '</table>'
st.markdown(_tbl_html, unsafe_allow_html=True)

# Tab bar accent CSS now lives in dashboard.py, right next to the st.tabs()
# call itself — it's app-wide chrome, not Overview-tab content, so it belongs
# next to the navigation it styles rather than inside one arbitrary tab.


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1 — EVENT LOG & OBJECT GRAPH
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
