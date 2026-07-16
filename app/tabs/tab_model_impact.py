# ── AI CAUSAL INTERPRETATION ─────────────────────────────────────────────
# Reuses the _ai_card/_ai_status component defined in tab_overview.py (which
# execs first — see dashboard.py's tab loop) so all three "AI ___ Summary"
# cards share one design instead of three hand-rolled variants. Content is
# derived from this run's actual do_result / ablation numbers rather than a
# fixed narrative, so it changes with the data instead of reading the same
# regardless of what was discovered.
_mi_wodk = ablation.get("without_domain_knowledge", {}) if ablation else {}
_mi_prec = _mi_wodk.get("precision", dag_metrics.get("precision", 0.0))
_mi_rec  = _mi_wodk.get("recall",    dag_metrics.get("recall",    0.0))
_mi_f1   = _mi_wodk.get("f1_score",  dag_metrics.get("f1_score",  0.0))
_mi_conf_col, _mi_conf_lbl = _ai_status((_mi_prec + _mi_rec + _mi_f1) / 3)

_mi_has_do = stage_status.get("do_operator") == "ok" and bool(do_result)
if _mi_has_do:
    _mi_causal  = do_result.get("causal", 0.0)
    _mi_naive   = do_result.get("naive", 0.0)
    _mi_gap_pct = do_result.get("gap_pct", 0.0)
    _mi_ci_low  = do_result.get("ci_low", 0.0)
    _mi_ci_high = do_result.get("ci_high", 0.0)
    _mi_method  = do_result.get("method_label", "Double ML")
    _mi_lead = (
        f"Confounding adjustment recovered a causal effect of <b>{_mi_causal:.2f} days</b> — "
        f"{abs(_mi_gap_pct):.0f}% different from the naive, uncorrected estimate."
    )
    _mi_bullet1 = (
        f"Naive correlation suggested <b>{_mi_naive:.2f} days</b>; {_mi_method} correction gives "
        f"<b>{_mi_causal:.2f} days</b> (95% CI [{_mi_ci_low:.2f}, {_mi_ci_high:.2f}])"
    )
else:
    _mi_lead = "The Structural Causal Model quantified the strength of every discovered causal link."
    _mi_bullet1 = "Run the Double ML estimator above to recover a confounding-adjusted causal effect"

st.markdown(
    _ai_card(
        accent="#8B5CF6", badge_bg="#F5F3FF", icon="✨", title="AI Causal Interpretation",
        conf_color=_mi_conf_col, conf_label=_mi_conf_lbl,
        lead=_mi_lead,
        bullets=[
            _mi_bullet1,
            f"Structural model validated at precision <b>{_mi_prec:.2f}</b> / recall "
            f"<b>{_mi_rec:.2f}</b> against planted ground truth",
            "Treatment effects vary by operational segment (CATE) — explore further down this tab",
            "Use the What-If Simulator below to test interventions, or the Case Inspector tab to "
            "drill into any individual case's outcome",
        ],
    ),
    unsafe_allow_html=True,
)

if is_custom:
    accuracy_disclaimer(custom_confidence, len(df), custom_quality.get("score", 0))

# ── SECTION 3: What-If Causal Simulator ────────────────────────────────────

# Session state initialisation
_is_mfg_sim = (domain == "manufacturing")
_sim_key    = "sim_levers_mfg" if _is_mfg_sim else "sim_levers_hc"

_MFG_DEFAULTS = {
    "supplier_reliability_pct": 40, "machine_capacity_expanded": False,
    "approval_automation": False, "additional_workforce": 0,
    "material_lead_time_mode": "Current", "carrier_express_pct": 15,
    "export_flag_reduction": False, "order_batching": False,
}
_HC_DEFAULTS = {
    "specialist_allocation_pct": 45, "bed_capacity_expanded": False,
    "triage_automation": False, "additional_nursing_staff": 0,
    "fast_track_eligibility_pct": 20, "diagnostic_speed_mode": "Standard",
}

if _sim_key not in st.session_state:
    st.session_state[_sim_key] = dict(_MFG_DEFAULTS if _is_mfg_sim else _HC_DEFAULTS)

# ── Causal engine ──────────────────────────────────────────────────────────
def _live_coef(parent, child, default):
    if not coefs.empty and "parent" in coefs.columns and "child" in coefs.columns:
        _m = coefs[(coefs["parent"] == parent) & (coefs["child"] == child)]
        if not _m.empty:
            _v = _m.iloc[0].get("estimated_value", default)
            if pd.notna(_v):
                return float(_v)
    return default

_sim_mfg_bl = round(float(df[cfg["outcome_var"]].mean()), 2) if cfg.get("outcome_var") in df.columns else 8.2
_sim_hc_bl  = round(float(df[cfg["outcome_var"]].mean()), 2) if cfg.get("outcome_var") in df.columns else 5.27

# Effect-size constants used both by the compute functions below and by the
# lever sliders' help= text further down, so the displayed claim can never
# silently drift from the coefficient actually applied.
_MFG_EFFECTS = {
    "export_reduction_pct":    35,    # exp_eff = -35% of BL_APD
    "machine_queue_units":     1.2,   # cap_eff — absolute units removed from queue
    "workforce_per_worker":    0.3,   # wf_eff per additional worker (units)
    "approval_automation_pct": 50,    # auto_eff = -50% of BL_APD
    "batching_days":           0.2,   # d_batch — flat delay reduction (days)
    "carrier_per_10pct_days":  0.08,  # d_carrier per +10% express carrier usage
}
_HC_EFFECTS = {
    "nurse_per_staff_days": 0.15,  # nurse_eff per additional nursing staff
    "triage_days":          0.30,  # triage_eff — flat delay reduction (days)
}


def _compute_mfg(levers):
    C_sup_mlt = _live_coef("supplier_a", "material_lead_time", 7.0)
    C_mlt_del = _live_coef("material_lead_time", "shipment_delay", 0.9)
    C_mql_apd = _live_coef("machine_queue_length", "approval_duration", 0.7)
    C_apd_del = _live_coef("approval_duration", "shipment_delay", 0.3)

    BL_DEL = _sim_mfg_bl;  BL_MLT = 7.2;  BL_MQL = 3.1;  BL_APD = 2.4
    SUP_BASE = 0.60;  CAR_BASE = 15

    sup_a      = 1.0 - levers["supplier_reliability_pct"] / 100.0
    mlt_factor = {"Current": 1.0, "Reduced (-20%)": 0.80, "Optimised (-40%)": 0.60}[
        levers["material_lead_time_mode"]]

    mlt_pre = BL_MLT + C_sup_mlt * (sup_a - SUP_BASE)
    mlt_val = mlt_pre * mlt_factor

    wf_eff  = -_MFG_EFFECTS["workforce_per_worker"] * levers["additional_workforce"]
    cap_eff = -_MFG_EFFECTS["machine_queue_units"] if levers["machine_capacity_expanded"] else 0.0
    mql_val = max(0.0, BL_MQL + wf_eff + cap_eff)

    exp_eff  = -_MFG_EFFECTS["export_reduction_pct"] / 100 * BL_APD if levers["export_flag_reduction"] else 0.0
    auto_eff = -_MFG_EFFECTS["approval_automation_pct"] / 100 * BL_APD if levers["approval_automation"] else 0.0
    q_eff    = C_mql_apd * (mql_val - BL_MQL)
    apd_val  = max(0.0, BL_APD + exp_eff + auto_eff + q_eff)

    d_sup     = C_mlt_del * C_sup_mlt * (sup_a - SUP_BASE)
    d_ltmode  = C_mlt_del * mlt_pre * (mlt_factor - 1.0)
    d_machine = C_apd_del * q_eff
    d_appr    = C_apd_del * (exp_eff + auto_eff)
    d_carrier = -_MFG_EFFECTS["carrier_per_10pct_days"] / 10 * (levers["carrier_express_pct"] - CAR_BASE)
    d_batch   = -_MFG_EFFECTS["batching_days"] if levers["order_batching"] else 0.0

    pred = max(0.5, BL_DEL + d_sup + d_ltmode + d_machine + d_appr + d_carrier + d_batch)
    imp  = (BL_DEL - pred) / BL_DEL * 100.0

    throughput = min(160.0, 100.0 * (1 + 0.3 * (1 - mql_val / BL_MQL)))
    risk_idx   = 45.0 * (pred / BL_DEL)

    impl_cost = (
        (50000 if levers["machine_capacity_expanded"] else 0) +
        (30000 if levers["approval_automation"]       else 0) +
        levers["additional_workforce"] * 5000 +
        max(0, levers["carrier_express_pct"] - CAR_BASE) * 200 +
        (20000 if levers["order_batching"]        else 0) +
        (15000 if levers["export_flag_reduction"] else 0)
    )
    # 300 orders/yr × $960/day logistics cost per delayed order (mid-size manufacturer)
    annual_sav = imp / 100.0 * BL_DEL * 300 * 960
    roi_mo     = (impl_cost / (annual_sav / 12)) if annual_sav > 0 else float("inf")

    return {
        "predicted": pred, "improvement_pct": imp,
        "mlt": mlt_val, "mql": mql_val, "apd": apd_val,
        "throughput": throughput, "risk_index": risk_idx,
        "impl_cost": impl_cost, "annual_saving": annual_sav, "roi_months": roi_mo,
        "ci_low": pred * 0.88, "ci_high": pred * 1.12,  # ±12% scenario uncertainty band
        "deltas": {
            "Supplier Change":     d_sup,
            "Lead Time Mode":      d_ltmode,
            "Machine & Workforce": d_machine,
            "Approval Actions":    d_appr,
            "Express Carrier":     d_carrier,
            "Order Batching":      d_batch,
        },
        "baseline": BL_DEL,
        "mediators": {
            "Material Lead Time":   (BL_MLT, mlt_val, "days"),
            "Machine Queue Length": (BL_MQL, mql_val, "units"),
            "Approval Duration":    (BL_APD, apd_val, "days"),
        },
    }

def _compute_hc(levers):
    BL = _sim_hc_bl;  BL_BED = 78.0;  BL_SPEC = 0.45;  FAST_BASE = 20

    spec_prob   = levers["specialist_allocation_pct"] / 100.0
    diag_factor = {"Standard": 1.0, "Fast (-20%)": 0.80, "Express (-35%)": 0.65}[
        levers["diagnostic_speed_mode"]]

    bed_eff    = -0.4 if levers["bed_capacity_expanded"] else 0.0
    nurse_eff  = -_HC_EFFECTS["nurse_per_staff_days"] * levers["additional_nursing_staff"]
    triage_eff = -_HC_EFFECTS["triage_days"] if levers["triage_automation"] else 0.0
    fast_eff   = -0.025 * (levers["fast_track_eligibility_pct"] - FAST_BASE)

    d_spec    = 1.8  * (spec_prob - BL_SPEC)
    d_diag    = BL   * 0.5 * (diag_factor - 1.0)
    d_bed     = 0.4  * bed_eff
    d_nursing = 0.4  * nurse_eff
    d_triage  = triage_eff
    d_fast    = fast_eff

    pred = max(0.5, BL + d_spec + d_diag + d_bed + d_nursing + d_triage + d_fast)
    imp  = (BL - pred) / BL * 100.0

    impl_cost = (
        (40000 if levers["bed_capacity_expanded"] else 0) +
        (25000 if levers["triage_automation"]     else 0) +
        levers["additional_nursing_staff"] * 4500
    )
    # 400 admissions/yr × $1,050/day bed cost (mid-size hospital ward)
    annual_sav = imp / 100.0 * BL * 400 * 1050
    roi_mo     = (impl_cost / (annual_sav / 12)) if annual_sav > 0 else float("inf")

    return {
        "predicted": pred, "improvement_pct": imp,
        "mlt": pred, "mql": BL_BED * (1 + bed_eff / 100), "apd": BL * 0.4,
        "throughput": min(160.0, 100.0 * (1 - d_bed * 0.5)),
        "risk_index": 45.0 * (pred / BL),
        "impl_cost": impl_cost, "annual_saving": annual_sav, "roi_months": roi_mo,
        "ci_low": pred * 0.88, "ci_high": pred * 1.12,  # ±12% scenario uncertainty band
        "deltas": {
            "Specialist Allocation": d_spec,
            "Diagnostic Speed":      d_diag,
            "Bed Capacity":          d_bed,
            "Nursing Staff":         d_nursing,
            "Triage Automation":     d_triage,
            "Fast Track":            d_fast,
        },
        "baseline": BL,
        "mediators": {
            "Length of Stay":      (BL,           pred,                      "days"),
            "Bed Occupancy":       (BL_BED,       BL_BED * (1 + bed_eff / 100), "%"),
            "Specialist Assigned": (BL_SPEC * 100, spec_prob * 100,           "%"),
        },
    }

# ── Recommended Action Plan — grid-search the lever space for the
# cheapest combination that reaches a target improvement, using the same
# _compute_mfg/_compute_hc engine that drives the rest of the simulator
# (not a separate model, so the recommendation and the live outcome
# numbers can never disagree). Small discrete grids keep this to a few
# thousand pure-arithmetic evaluations — fast enough to run on every
# rerun without caching.
_MFG_GRID = {
    "supplier_reliability_pct": [40, 60, 80, 100],
    "machine_capacity_expanded": [False, True],
    "approval_automation": [False, True],
    "additional_workforce": [0, 5, 10, 15, 20],
    "material_lead_time_mode": ["Current", "Reduced (-20%)", "Optimised (-40%)"],
    "carrier_express_pct": [15, 35, 55, 75, 95],
    "export_flag_reduction": [False, True],
    "order_batching": [False, True],
}
_HC_GRID = {
    "specialist_allocation_pct": [45, 65, 85, 100],
    "bed_capacity_expanded": [False, True],
    "triage_automation": [False, True],
    "additional_nursing_staff": [0, 5, 10, 15, 20],
    "fast_track_eligibility_pct": [20, 40, 60, 80, 100],
    "diagnostic_speed_mode": ["Standard", "Fast (-20%)", "Express (-35%)"],
}

def _recommend_plan(target_pct, defaults, grid, compute_fn):
    import itertools
    _keys = list(grid.keys())
    _cheapest, _best_any = None, None
    for _combo in itertools.product(*[grid[k] for k in _keys]):
        _levers = dict(defaults)
        _levers.update(dict(zip(_keys, _combo)))
        _r = compute_fn(_levers)
        if _best_any is None or _r["improvement_pct"] > _best_any[1]["improvement_pct"]:
            _best_any = (_levers, _r)
        if _r["improvement_pct"] >= target_pct:
            if _cheapest is None or _r["impl_cost"] < _cheapest[1]["impl_cost"]:
                _cheapest = (_levers, _r)
    if _cheapest is not None:
        return _cheapest[0], _cheapest[1], True
    return _best_any[0], _best_any[1], False

# ── Header banner ──────────────────────────────────────────────────────────
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

_sim_domain_lbl = ("Manufacturing — Prihir Enterprises"
                   if _is_mfg_sim else "Healthcare — Hospital Admissions")
_sim_bl      = _sim_mfg_bl if _is_mfg_sim else _sim_hc_bl
_sim_out_lbl = "Shipment Delay" if _is_mfg_sim else "Treatment Duration"
_sim_f1      = dag_metrics.get("f1_score", 0.0) if "dag_metrics" in dir() else 0.0
_conf_lbl    = "HIGH" if _sim_f1 >= 0.9 else ("MODERATE" if _sim_f1 >= 0.7 else "LOW")
_conf_col    = "#059669" if _sim_f1 >= 0.9 else ("#D97706" if _sim_f1 >= 0.7 else "#DC2626")

def _sim_stat_card(icon_bg, icon, label, value, value_color="#0F172A"):
    return (
        f'<div style="display:flex;align-items:center;gap:10px;background:#F8FAFC;'
        f'border:1px solid #E2E8F0;border-radius:12px;padding:10px 14px;min-width:150px;">'
        f'<div style="width:34px;height:34px;border-radius:9px;background:{icon_bg};flex-shrink:0;'
        f'display:flex;align-items:center;justify-content:center;font-size:0.95rem;">{icon}</div>'
        f'<div><div style="font-size:0.62rem;color:#94A3B8;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.04em;">{label}</div>'
        f'<div style="font-size:0.85rem;font-weight:700;color:{value_color};">{value}</div></div>'
        f'</div>'
    )

st.markdown(
    f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:16px;'
    f'padding:20px 24px;margin-bottom:20px;'
    f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;'
    f'box-shadow:0 2px 10px rgba(15,23,42,0.03);">'
    f'<div style="display:flex;align-items:center;gap:16px;">'
    f'<div style="width:48px;height:48px;border-radius:14px;background:#EDE9FE;flex-shrink:0;'
    f'display:flex;align-items:center;justify-content:center;font-size:1.3rem;">⚡</div>'
    f'<div>'
    f'<div style="font-size:1.25rem;font-weight:800;color:#1E293B;margin-bottom:3px;">'
    f'What-If Causal Simulator</div>'
    f'<div style="font-size:0.85rem;color:#64748B;">Adjust levers to see real-time outcome '
    f'predictions from the discovered causal model.</div>'
    f'</div>'
    f'</div>'
    f'<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">'
    + _sim_stat_card("#DBEAFE", "📁", "Domain", _sim_domain_lbl)
    + _sim_stat_card("#EDE9FE", "📅", f"Baseline {_sim_out_lbl}", f"{_sim_bl:.2f} days")
    + _sim_stat_card("#D1FAE5", "🛡️", "Model Confidence", f"{_conf_lbl} Confidence", _conf_col)
    + f'<div style="display:flex;align-items:center;gap:6px;background:#ECFDF5;border:1px solid #A7F3D0;'
    f'border-radius:20px;padding:8px 16px;">'
    f'<span style="width:8px;height:8px;background:#10B981;border-radius:50%;'
    f'box-shadow:0 0 0 3px rgba(16,185,129,0.22);display:inline-block;"></span>'
    f'<span style="font-size:0.78rem;font-weight:600;color:#059669;">Live Engine Active</span>'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

def _sim_section_header(icon_bg, icon, title):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;">'
        f'<div style="width:26px;height:26px;border-radius:7px;background:{icon_bg};'
        f'display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">{icon}</div>'
        f'<span style="color:#1E293B;font-size:0.95rem;font-weight:700;">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Two-column layout ──────────────────────────────────────────────────────
_lcol, _rcol = st.columns([1.1, 1.9])

with _lcol:
    _sim_section_header("#EDE9FE", "🎛️", "Intervention Levers")

    if st.button("🔄 Reset to Baseline", use_container_width=True, key="sim_reset_btn"):
        st.session_state[_sim_key] = dict(_MFG_DEFAULTS if _is_mfg_sim else _HC_DEFAULTS)
        st.rerun()

    _lv = st.session_state[_sim_key]

    if _is_mfg_sim:
        with st.expander("🏭 Supplier & Procurement", expanded=True):
            _lv["supplier_reliability_pct"] = st.slider(
                "Supplier B Allocation (%)", 0, 100,
                _lv["supplier_reliability_pct"], 5, key="lv_sup_rel",
                help="60% currently go to Supplier A (unreliable). Shift more to Supplier B.",
            )
            _lv["export_flag_reduction"] = st.toggle(
                "Streamline Export Documentation",
                _lv["export_flag_reduction"], key="lv_exp_flag",
                help=f"Reduces export-related approval delays ~{_MFG_EFFECTS['export_reduction_pct']}%",
            )
        with st.expander("⚙️ Machine & Capacity", expanded=True):
            _lv["machine_capacity_expanded"] = st.toggle(
                "Expand Machine Capacity",
                _lv["machine_capacity_expanded"], key="lv_mach_cap",
                help=f"Adds processing units, reduces Machine Queue by up to {_MFG_EFFECTS['machine_queue_units']} units",
            )
            _lv["additional_workforce"] = st.slider(
                "Additional Workforce", 0, 20,
                _lv["additional_workforce"], 1, key="lv_workforce",
                help=f"Each additional worker reduces queue ~{_MFG_EFFECTS['workforce_per_worker']} units",
            )
        with st.expander("📋 Approvals & Process", expanded=False):
            _lv["approval_automation"] = st.toggle(
                "Automate Approval Steps",
                _lv["approval_automation"], key="lv_appr_auto",
                help=f"Reduces Approval Duration ~{_MFG_EFFECTS['approval_automation_pct']}%",
            )
            _lv["order_batching"] = st.toggle(
                "Enable Order Batching",
                _lv["order_batching"], key="lv_batching",
                help=f"Reduces delay by ~{_MFG_EFFECTS['batching_days']} days",
            )
        with st.expander("🚚 Logistics & Delivery", expanded=False):
            _lv["carrier_express_pct"] = st.slider(
                "Express Carrier Usage (%)", 0, 100,
                _lv["carrier_express_pct"], 5, key="lv_carrier",
                help=f"Each 10% increase reduces delay ~{_MFG_EFFECTS['carrier_per_10pct_days']} days",
            )
            _lv["material_lead_time_mode"] = st.radio(
                "Material Lead Time Strategy",
                ["Current", "Reduced (-20%)", "Optimised (-40%)"],
                index=["Current", "Reduced (-20%)", "Optimised (-40%)"].index(
                    _lv["material_lead_time_mode"]),
                key="lv_lt_mode",
                help="Negotiate faster material delivery contracts",
            )
    else:
        with st.expander("🏥 Specialist & Allocation", expanded=True):
            _lv["specialist_allocation_pct"] = st.slider(
                "Specialist Allocation (%)", 0, 100,
                _lv["specialist_allocation_pct"], 5, key="lv_spec_alloc",
                help="Baseline: 45% of cases assigned a specialist",
            )
            _lv["fast_track_eligibility_pct"] = st.slider(
                "Fast Track Eligibility (%)", 0, 100,
                _lv["fast_track_eligibility_pct"], 5, key="lv_fast_track",
                help="Baseline: 20% on fast-track pathway",
            )
        with st.expander("🛏️ Capacity & Staff", expanded=True):
            _lv["bed_capacity_expanded"] = st.toggle(
                "Expand Bed Capacity",
                _lv["bed_capacity_expanded"], key="lv_bed_cap",
                help="Adds beds, reduces occupancy pressure",
            )
            _lv["additional_nursing_staff"] = st.slider(
                "Additional Nursing Staff", 0, 20,
                _lv["additional_nursing_staff"], 1, key="lv_nursing",
                help=f"Each nurse reduces duration ~{_HC_EFFECTS['nurse_per_staff_days']} days",
            )
        with st.expander("⚡ Process & Diagnostics", expanded=False):
            _lv["triage_automation"] = st.toggle(
                "Automate Triage Scoring",
                _lv["triage_automation"], key="lv_triage_auto",
                help=f"Reduces triage-to-treatment delay ~{_HC_EFFECTS['triage_days']} days",
            )
            _lv["diagnostic_speed_mode"] = st.radio(
                "Diagnostic Speed Mode",
                ["Standard", "Fast (-20%)", "Express (-35%)"],
                index=["Standard", "Fast (-20%)", "Express (-35%)"].index(
                    _lv["diagnostic_speed_mode"]),
                key="lv_diag_speed",
            )

    # Active-intervention summary panel
    _defaults = _MFG_DEFAULTS if _is_mfg_sim else _HC_DEFAULTS
    _changes  = [(k, _defaults[k], v) for k, v in _lv.items() if v != _defaults[k]]
    if _changes:
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        _chg_html = (
            f'<div style="background:#F8FAFC;border:1px solid {BORDER};'
            f'border-radius:8px;padding:10px 12px;">'
            f'<div style="font-size:0.7rem;font-weight:700;color:{MUTED};'
            f'text-transform:uppercase;margin-bottom:5px;">Active Interventions</div>'
        )
        for _k, _bv, _cv in _changes:
            _klbl = _k.replace("_", " ").title()
            _bstr = (f"{_bv}%" if isinstance(_bv, int) and "pct" in _k
                     else ("ON" if _bv is True else ("OFF" if _bv is False else str(_bv))))
            _cstr = (f"{_cv}%" if isinstance(_cv, int) and "pct" in _k
                     else ("ON" if _cv is True else ("OFF" if _cv is False else str(_cv))))
            _chg_html += (
                f'<div style="font-size:0.78rem;color:#059669;margin-bottom:1px;">'
                f'✓ {_klbl}: {_bstr} → {_cstr}</div>'
            )
        _chg_html += "</div>"
        st.markdown(_chg_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;'
            f'padding:8px 12px;font-size:0.78rem;color:{MUTED};margin-top:6px;">'
            f'All levers at baseline — adjust above to simulate.</div>',
            unsafe_allow_html=True,
        )

# Compute predictions from current lever state
_res = _compute_mfg(_lv) if _is_mfg_sim else _compute_hc(_lv)

with _rcol:
    _sim_section_header("#DBEAFE", "📊", "Predicted Outcomes")

    # ── Hero KPI + impact badge ───────────────────────────────────────────
    _imp  = _res["improvement_pct"]
    _pred = _res["predicted"]
    _bl   = _res["baseline"]

    if _imp > 25:
        _badge_col, _badge_lbl = "#059669", "HIGH IMPACT"
    elif _imp > 10:
        _badge_col, _badge_lbl = "#D97706", "MODERATE IMPACT"
    elif _imp > 0:
        _badge_col, _badge_lbl = "#EA580C", "LOW IMPACT"
    else:
        _badge_col, _badge_lbl = "#DC2626", "NO IMPROVEMENT"

    with st.container(border=True):
        _hc1, _hc2 = st.columns([2, 1])
        with _hc1:
            st.metric(
                label=f"Predicted {_sim_out_lbl}",
                value=f"{_pred:.1f} days",
                delta=f"{-_imp:.1f}% vs baseline",
                delta_color="inverse",
            )
            st.markdown(
                f'<div style="font-size:0.75rem;color:{MUTED};font-style:italic;margin-top:-6px;">'
                f'Scenario range (±12%): [{_res["ci_low"]:.1f} – {_res["ci_high"]:.1f} days]</div>',
                unsafe_allow_html=True,
            )
        with _hc2:
            st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:{_badge_col};color:#fff;border-radius:8px;'
                f'padding:7px 10px;font-size:0.72rem;font-weight:700;text-align:center;">'
                f'{_badge_lbl}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── Secondary KPI row ─────────────────────────────────────────────────
    _k1, _k2, _k3 = st.columns(3)
    with _k1:
        with st.container(border=True):
            st.metric("Throughput", f"{_res['throughput']:.0f}/day",
                      delta=f"+{_res['throughput'] - 100:.0f}")
    with _k2:
        with st.container(border=True):
            st.metric("Risk Index", f"{_res['risk_index']:.1f} / 100",
                      delta=f"{_res['risk_index'] - 45:.1f} vs baseline",
                      delta_color="inverse",
                      help="Composite risk score (0–100). Baseline = 45. Lower is better. Scales with predicted delay relative to baseline.")
    with _k3:
        with st.container(border=True):
            _roi     = _res["roi_months"]
            _no_sav  = _res["annual_saving"] <= 0
            _roi_str = "—" if _no_sav else (f"{_roi:.1f} mo" if _roi < 60 else ">5 yr")
            st.metric("ROI Payback", _roi_str,
                      delta="Adjust levers above" if _no_sav else f"${_res['annual_saving']:,.0f}/yr saved",
                      delta_color="off" if _no_sav else "normal")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── Causal Effect Decomposition (waterfall) ───────────────────────────
    _sim_wf_container = st.container(border=True)
    with _sim_wf_container:
        _sim_section_header("#FEF3C7", "📡", "Causal Effect Decomposition")
    _wf_labels  = [f"Baseline ({_bl:.1f}d)"] + list(_res["deltas"].keys()) + ["Predicted"]
    _wf_values  = [_bl] + list(_res["deltas"].values()) + [_pred]
    _wf_measure = ["absolute"] + ["relative"] * len(_res["deltas"]) + ["total"]

    _fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=_wf_measure,
        x=_wf_labels,
        y=_wf_values,
        connector={"line": {"color": BORDER, "width": 1, "dash": "dot"}},
        decreasing={"marker": {"color": SUCCESS}},
        increasing={"marker": {"color": ERROR}},
        totals={"marker": {"color": INFO}},
        text=[
            f"{v:+.2f}d" if i not in (0, len(_wf_values) - 1) else f"{v:.1f}d"
            for i, v in enumerate(_wf_values)
        ],
        textposition="outside",
        textfont={"size": 10},
    ))
    _wfl = dict(**PLOTLY_LAYOUT)
    _wfl.update(dict(
        height=300,
        margin=dict(l=10, r=10, t=20, b=50),
        yaxis={"title": "Days", "gridcolor": BORDER, "tickformat": ".1f"},
        xaxis={"tickfont": {"size": 9}},
        showlegend=False,
    ))
    _fig_wf.update_layout(**_wfl)
    with _sim_wf_container:
        st.plotly_chart(_fig_wf, use_container_width=True, theme=None,
                        config={"displayModeBar": False})

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Mediator states + Business impact (full-width two-column row) ─────────
_medcol, _biscol = st.columns([1, 1])
with _medcol:
    with st.container(border=True):
        _sim_section_header("#DBEAFE", "🔗", "Mediator Variable States")
        _med_html = (
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;font-size:0.8rem;">'
            '<thead><tr>'
            f'<th style="padding:7px 8px;text-align:left;border-bottom:2px solid {BORDER};'
            f'color:{MUTED};font-weight:600;">Variable</th>'
            f'<th style="padding:7px 8px;text-align:right;border-bottom:2px solid {BORDER};'
            f'color:{MUTED};font-weight:600;">Baseline</th>'
            f'<th style="padding:7px 8px;text-align:right;border-bottom:2px solid {BORDER};'
            f'color:{MUTED};font-weight:600;">Predicted</th>'
            f'<th style="padding:7px 8px;text-align:right;border-bottom:2px solid {BORDER};'
            f'color:{MUTED};font-weight:600;">Change</th>'
            '</tr></thead><tbody>'
        )
        for _var, (_bv, _cv, _unit) in _res["mediators"].items():
            _delta = _cv - _bv
            _dstr  = f"{_delta:+.2f} {_unit}"
            _dcol  = (SUCCESS if _delta < -0.001 else (ERROR if _delta > 0.001 else MUTED))
            _med_html += (
                f'<tr style="border-bottom:1px solid {BORDER};">'
                f'<td style="padding:6px 8px;color:{TEXT};font-weight:500;">{_var}</td>'
                f'<td style="padding:6px 8px;text-align:right;color:{MUTED};">{_bv:.2f} {_unit}</td>'
                f'<td style="padding:6px 8px;text-align:right;color:{TEXT};font-weight:600;">{_cv:.2f} {_unit}</td>'
                f'<td style="padding:6px 8px;text-align:right;font-weight:700;color:{_dcol};">{_dstr}</td>'
                f'</tr>'
            )
        _med_html += "</tbody></table></div>"
        st.markdown(_med_html, unsafe_allow_html=True)

with _biscol:
    with st.container(border=True):
        _sim_section_header("#DBEAFE", "💰", "Business Impact Estimate")
        if _res["impl_cost"] > 0:
            st.markdown(
                f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">'
                f'<div><div style="font-size:0.68rem;color:{MUTED};font-weight:600;'
                f'text-transform:uppercase;">Impl. Cost</div>'
                f'<div style="font-size:0.95rem;font-weight:700;color:{TEXT};">'
                f'${_res["impl_cost"]:,.0f}</div></div>'
                f'<div><div style="font-size:0.68rem;color:{MUTED};font-weight:600;'
                f'text-transform:uppercase;">Annual Savings</div>'
                f'<div style="font-size:0.95rem;font-weight:700;color:{SUCCESS};">'
                f'${_res["annual_saving"]:,.0f}/yr</div></div>'
                f'<div><div style="font-size:0.68rem;color:{MUTED};font-weight:600;'
                f'text-transform:uppercase;">Payback</div>'
                f'<div style="font-size:0.95rem;font-weight:700;color:{TEXT};">{_roi_str}</div></div>'
                f'</div>'
                f'<div style="font-size:0.68rem;color:{SUBTLE};margin-top:8px;font-style:italic;">'
                + (f'*SEM-based estimate · $960/day/order · 300 orders/yr'
                   if _is_mfg_sim else
                   f'*SEM-based estimate · $1,050/day/case · 400 cases/yr')
                + f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="color:{MUTED};font-size:0.82rem;">'
                f'No implementation cost at current lever settings.</div>',
                unsafe_allow_html=True,
            )

# ── Scenario Comparison + Live Causal Graph (final two-column row) ────────
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
_botl, _botr = st.columns([1, 1])

if "sim_target_pct" not in st.session_state:
    st.session_state["sim_target_pct"] = 20

with _botl:
    with st.container(border=True):
        _sim_section_header("#FEF3C7", "🎯", "Recommended Action Plan")

        st.markdown(
            f'<div style="font-size:0.75rem;color:{MUTED};margin-bottom:6px;">'
            f'Cheapest lever combination that reaches your target reduction — searched '
            f'across the same causal engine driving the numbers above, so it never disagrees '
            f'with what you see there.</div>',
            unsafe_allow_html=True,
        )
        _target_pct = st.slider(
            "Target delay reduction (%)", 5, 50,
            st.session_state["sim_target_pct"], 5, key="sim_target_slider",
        )
        st.session_state["sim_target_pct"] = _target_pct

        _plan_defaults = _MFG_DEFAULTS if _is_mfg_sim else _HC_DEFAULTS
        _plan_grid     = _MFG_GRID if _is_mfg_sim else _HC_GRID
        _plan_compute  = _compute_mfg if _is_mfg_sim else _compute_hc
        _plan_levers, _plan_res, _plan_hit = _recommend_plan(
            _target_pct, _plan_defaults, _plan_grid, _plan_compute)

        if _plan_hit:
            st.markdown(
                f'<div style="background:#ECFDF5;border:1px solid #A7F3D0;border-radius:10px;'
                f'padding:12px 14px;margin:8px 0;">'
                f'<div style="font-size:0.78rem;color:#059669;font-weight:800;text-transform:uppercase;'
                f'letter-spacing:0.04em;margin-bottom:4px;">Target reachable</div>'
                f'<div style="font-size:0.85rem;color:{TEXT};">Predicted {_sim_out_lbl.lower()}: '
                f'<b>{_plan_res["predicted"]:.1f} days</b> '
                f'(<span style="color:#059669;font-weight:700;">-{_plan_res["improvement_pct"]:.0f}%</span> '
                f'vs {_plan_res["baseline"]:.1f}d baseline)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;'
                f'padding:12px 14px;margin:8px 0;">'
                f'<div style="font-size:0.78rem;color:#D97706;font-weight:800;text-transform:uppercase;'
                f'letter-spacing:0.04em;margin-bottom:4px;">Target not reachable with these levers</div>'
                f'<div style="font-size:0.85rem;color:{TEXT};">Best achievable: '
                f'<b>{_plan_res["predicted"]:.1f} days</b> '
                f'(<span style="color:#D97706;font-weight:700;">-{_plan_res["improvement_pct"]:.0f}%</span> '
                f'vs {_plan_res["baseline"]:.1f}d baseline) — shown below anyway.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _plan_changes = [(k, _plan_defaults[k], v) for k, v in _plan_levers.items()
                          if v != _plan_defaults[k]]
        if _plan_changes:
            _plan_html = (
                f'<div style="background:#F8FAFC;border:1px solid {BORDER};border-radius:8px;'
                f'padding:10px 12px;margin-bottom:10px;">'
                f'<div style="font-size:0.7rem;font-weight:700;color:{MUTED};'
                f'text-transform:uppercase;margin-bottom:5px;">Recommended Levers</div>'
            )
            for _k, _bv, _cv in _plan_changes:
                _klbl = _k.replace("_", " ").title()
                _cstr = (f"{_cv}%" if isinstance(_cv, int) and "pct" in _k
                         else ("ON" if _cv is True else ("OFF" if _cv is False else str(_cv))))
                _plan_html += (
                    f'<div style="font-size:0.78rem;color:#334155;margin-bottom:1px;">'
                    f'&#9679; {_klbl}: <b>{_cstr}</b></div>'
                )
            _plan_html += "</div>"
            st.markdown(_plan_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="color:{MUTED};font-size:0.8rem;margin-bottom:10px;">'
                f'Baseline already meets this target.</div>',
                unsafe_allow_html=True,
            )

        _pk1, _pk2, _pk3 = st.columns(3)
        with _pk1:
            st.metric("Impl. Cost", f"${_plan_res['impl_cost']:,.0f}")
        with _pk2:
            st.metric("Annual Savings", f"${_plan_res['annual_saving']:,.0f}")
        with _pk3:
            _plan_roi = _plan_res["roi_months"]
            st.metric("Payback", "—" if _plan_res["annual_saving"] <= 0
                       else (f"{_plan_roi:.1f} mo" if _plan_roi < 60 else ">5 yr"))

        if st.button("✅ Apply This Plan to Levers", use_container_width=True, key="sim_apply_plan_btn"):
            st.session_state[_sim_key] = dict(_plan_levers)
            st.rerun()

        _plan_fig = go.Figure(go.Bar(
            x=["Baseline", "Recommended Plan"],
            y=[_plan_res["baseline"], _plan_res["predicted"]],
            marker_color=[MUTED, SUCCESS if _plan_hit else WARNING],
            text=[f"{_plan_res['baseline']:.1f}d", f"{_plan_res['predicted']:.1f}d"],
            textposition="outside",
        ))
        _plan_fig_layout = dict(**PLOTLY_LAYOUT)
        _plan_fig_layout.update(dict(
            height=200, showlegend=False,
            margin=dict(l=10, r=10, t=20, b=20),
            yaxis={"title": f"{_sim_out_lbl} (days)", "gridcolor": BORDER},
        ))
        _plan_fig.update_layout(**_plan_fig_layout)
        st.plotly_chart(_plan_fig, use_container_width=True, theme=None,
                        config={"displayModeBar": False})

with _botr:
    with st.container(border=True):
        _sim_lg1, _sim_lg2 = st.columns([5, 1])
        with _sim_lg1:
            _sim_section_header("#D1FAE5", "🔀", "Live Causal Graph — Intervention Propagation")
        with _sim_lg2:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:5px;margin-top:4px;">'
                '<span style="width:7px;height:7px;background:#10B981;border-radius:50%;'
                'box-shadow:0 0 0 3px rgba(16,185,129,0.22);display:inline-block;"></span>'
                '<span style="font-size:0.72rem;font-weight:600;color:#059669;">Live</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Lever "strength" — how far a lever sits from its baseline,
        # normalized to [0, 1] by its own plausible range. Drives edge
        # thickness/color/glow below so the graph reads impact magnitude,
        # not just an on/off toggle.
        def _lstr(key):
            _val, _def = _lv[key], _defaults[key]
            if isinstance(_val, bool):
                return 1.0 if _val != _def else 0.0
            if isinstance(_val, (int, float)):
                if "pct" in key:
                    _rng = 100.0
                elif key in ("additional_workforce", "additional_nursing_staff"):
                    _rng = 20.0
                else:
                    _rng = max(abs(_val), abs(_def), 1.0)
                return min(1.0, abs(_val - _def) / _rng)
            _cat_opts = {
                "material_lead_time_mode": ["Current", "Reduced (-20%)", "Optimised (-40%)"],
                "diagnostic_speed_mode":    ["Standard", "Fast (-20%)", "Express (-35%)"],
            }
            if key in _cat_opts:
                _opts = _cat_opts[key]
                return abs(_opts.index(_val) - _opts.index(_def)) / (len(_opts) - 1)
            return 0.0

        # Human-readable "what's driving this edge" line for hover tooltips.
        def _lever_desc(key, label):
            _val, _def = _lv[key], _defaults[key]
            if _val == _def:
                return None
            if isinstance(_val, bool):
                _vs = "ON" if _val else "OFF"
            elif isinstance(_val, (int, float)) and "pct" in key:
                _vs = f"{_val}%"
            else:
                _vs = str(_val)
            return f"{label}: {_vs}"

        if _is_mfg_sim:
            _dag_nodes = [
                {"id": "order_complexity",      "x": 0.0, "y": 3.0, "role": "Confounder"},
                {"id": "supplier_a",            "x": 0.0, "y": 0.0, "role": "Treatment"},
                {"id": "material_lead_time",    "x": 2.5, "y": 0.0, "role": "Mediator"},
                {"id": "machine_queue",         "x": 2.5, "y": 3.0, "role": "Mediator"},
                {"id": "approval_duration",     "x": 5.0, "y": 2.0, "role": "Mediator"},
                {"id": "export_flag",           "x": 2.5, "y": 5.0, "role": "Confounder"},
                {"id": "carrier_express",       "x": 5.0, "y": 0.0, "role": "Treatment"},
                {"id": "shipment_delay",        "x": 7.5, "y": 2.0, "role": "Outcome"},
            ]
            _node_val = {
                "material_lead_time": _res["mediators"]["Material Lead Time"],
                "machine_queue":      _res["mediators"]["Machine Queue Length"],
                "approval_duration":  _res["mediators"]["Approval Duration"],
                "shipment_delay":     (_bl, _pred, "days"),
            }
            # Nodes with no tracked SCM value of their own (pure lever
            # settings, not a mediator the model computes) fall back to how
            # far their driving lever sits from baseline.
            _root_lever = {
                "supplier_a":      "supplier_reliability_pct",
                "export_flag":     "export_flag_reduction",
                "carrier_express": "carrier_express_pct",
            }
            _drivers = {
                "supplier_a":      [_lever_desc("supplier_reliability_pct", "Supplier B Allocation")],
                "export_flag":     [_lever_desc("export_flag_reduction", "Export Docs Streamlined")],
                "carrier_express": [_lever_desc("carrier_express_pct", "Express Carrier Usage")],
                "machine_queue":   [_lever_desc("machine_capacity_expanded", "Machine Capacity Expanded"),
                                     _lever_desc("additional_workforce", "Additional Workforce")],
                "approval_duration": [_lever_desc("approval_automation", "Approval Automation"),
                                       _lever_desc("export_flag_reduction", "Export Docs Streamlined")],
                "material_lead_time": [_lever_desc("material_lead_time_mode", "Lead Time Strategy"),
                                        _lever_desc("supplier_reliability_pct", "Supplier B Allocation")],
            }
            _dag_edge_pairs = [
                ("supplier_a",         "material_lead_time"),
                ("order_complexity",   "machine_queue"),
                ("machine_queue",      "approval_duration"),
                ("export_flag",        "approval_duration"),
                ("material_lead_time", "shipment_delay"),
                ("approval_duration",  "shipment_delay"),
                ("carrier_express",    "shipment_delay"),
                ("order_complexity",   "shipment_delay"),
            ]
        else:
            _dag_nodes = [
                {"id": "patient_complexity",     "x": 0.0, "y": 1.0, "role": "Confounder"},
                {"id": "specialist_requirement", "x": 0.0, "y": 3.0, "role": "Treatment"},
                {"id": "bed_occupancy",          "x": 3.0, "y": 2.0, "role": "Mediator"},
                {"id": "triage_score",           "x": 1.5, "y": 0.0, "role": "Mediator"},
                {"id": "treatment_duration",     "x": 6.0, "y": 2.0, "role": "Outcome"},
            ]
            _node_val = {
                "specialist_requirement": _res["mediators"]["Specialist Assigned"],
                "bed_occupancy":          _res["mediators"]["Bed Occupancy"],
                "treatment_duration":     (_bl, _pred, "days"),
            }
            _root_lever = {
                "triage_score": "triage_automation",
            }
            _drivers = {
                "specialist_requirement": [_lever_desc("specialist_allocation_pct", "Specialist Allocation")],
                "bed_occupancy":          [_lever_desc("bed_capacity_expanded", "Bed Capacity Expanded"),
                                            _lever_desc("additional_nursing_staff", "Additional Nursing Staff")],
                "triage_score":           [_lever_desc("triage_automation", "Triage Automation")],
            }
            _dag_edge_pairs = [
                ("patient_complexity",     "treatment_duration"),
                ("specialist_requirement", "treatment_duration"),
                ("bed_occupancy",          "treatment_duration"),
                ("triage_score",           "specialist_requirement"),
            ]

        def _node_strength(nid):
            if nid in _node_val:
                _bv, _cv, _unit = _node_val[nid]
                _rng = max(abs(_bv), 0.5)
                return min(1.0, abs(_cv - _bv) / _rng)
            if nid in _root_lever:
                return _lstr(_root_lever[nid])
            return 0.0  # pure exogenous confounder — never intervened on here

        # Edge strength = how much the *source* node actually moved, so a
        # change made two hops upstream (e.g. more workforce → shorter
        # machine queue → shorter approval duration) correctly lights up
        # every downstream edge, not just the one tied to the lever you
        # touched directly.
        _dag_edges = [(_src, _dst, _node_strength(_src)) for _src, _dst in _dag_edge_pairs]

        _role_colors = {
            "Confounder": WARNING, "Treatment": PRIMARY, "Mediator": INFO, "Outcome": ERROR,
        }
        # Wider spacing (1.35x) so labels have room to breathe and don't
        # crowd/overlap neighboring nodes or edges.
        _space = 1.35
        for _n in _dag_nodes:
            _n["x"] *= _space
            _n["y"] *= _space
        _nxs = {n["id"]: n["x"] for n in _dag_nodes}
        _nys = {n["id"]: n["y"] for n in _dag_nodes}

        # Nodes touched by at least one currently-active edge get a soft
        # halo — real derived state (driven by the same strength values as
        # edge color/width), scaled by the strongest edge touching that
        # node, so it's obvious at a glance which part of the graph your
        # current lever changes are running through and how hard.
        _active_nodes = {}
        for _src, _dst, _strength in _dag_edges:
            if _strength > 0.001:
                _active_nodes[_src] = max(_active_nodes.get(_src, 0.0), _strength)
                _active_nodes[_dst] = max(_active_nodes.get(_dst, 0.0), _strength)

        def _lerp_color(c1, c2, t):
            t = max(0.0, min(1.0, t))
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
            return f"#{round(r1+(r2-r1)*t):02X}{round(g1+(g2-g1)*t):02X}{round(b1+(b2-b1)*t):02X}"

        import math
        _fig_dag = go.Figure()
        for _src, _dst, _strength in _dag_edges:
            _active = _strength > 0.001
            _x0, _y0, _x1, _y1 = _nxs[_src], _nys[_src], _nxs[_dst], _nys[_dst]
            _ecol = _lerp_color("#D7DEE8", PRIMARY, _strength)
            _ewid = 1.2 + _strength * 4.5
            _fig_dag.add_trace(go.Scatter(
                x=[_x0, _x1, None], y=[_y0, _y1, None],
                mode="lines",
                line=dict(color=_ecol, width=_ewid),
                showlegend=False, hoverinfo="skip",
            ))
            # Directional arrowhead ~62% of the way from source to target —
            # close enough to the destination to read as "pointing at it"
            # without the marker sitting on top of the node itself.
            _dx, _dy = _x1 - _x0, _y1 - _y0
            _ang = math.degrees(math.atan2(-_dy, _dx))  # plotly angle is clockwise from east
            _ax, _ay = _x0 + _dx * 0.62, _y0 + _dy * 0.62
            _fig_dag.add_trace(go.Scatter(
                x=[_ax], y=[_ay], mode="markers",
                marker=dict(symbol="triangle-right", size=9 + _strength * 4,
                            color=_ecol, angle=_ang, line=dict(width=0)),
                showlegend=False, hoverinfo="skip",
            ))
            # Invisible wide-hit marker at the edge midpoint carries the
            # hover tooltip — reports which lever(s) drive this edge and
            # how strong the current pull is, so hovering explains *why*
            # a segment lit up rather than just showing that it did.
            _mx, _my = (_x0 + _x1) / 2, (_y0 + _y1) / 2
            _dsrc = [d for d in _drivers.get(_src, []) if d]
            if not _dsrc:
                _dsrc = [d for d in _drivers.get(_dst, []) if d]
            if _dsrc:
                _drv_txt = "<br>".join(_dsrc)
            elif _strength > 0.001:
                _drv_txt = f"{_src.replace('_',' ').title()} moved from its baseline (propagated effect)"
            else:
                _drv_txt = "No active lever on this edge"
            _fig_dag.add_trace(go.Scatter(
                x=[_mx], y=[_my], mode="markers",
                marker=dict(size=22, color="rgba(0,0,0,0)"),
                showlegend=False,
                hovertemplate=(
                    f"<b>{_src.replace('_',' ').title()} → {_dst.replace('_',' ').title()}</b><br>"
                    f"Strength: {_strength*100:.0f}%<br>{_drv_txt}<extra></extra>"
                ),
            ))
        for _n in _dag_nodes:
            _ncol = _role_colors.get(_n["role"], MUTED)
            _nstr = _active_nodes.get(_n["id"], 0.0)
            if _nstr > 0.001:
                _fig_dag.add_trace(go.Scatter(
                    x=[_n["x"]], y=[_n["y"]], mode="markers",
                    marker=dict(size=54 + _nstr * 26, color=_ncol,
                                opacity=0.12 + _nstr * 0.16, line=dict(width=0)),
                    showlegend=False, hoverinfo="skip",
                ))
            _val_bits = ""
            if _n["id"] in _node_val:
                _bv, _cv, _unit = _node_val[_n["id"]]
                _val_bits = f"<br>{_bv:.2f} → {_cv:.2f} {_unit}"
            _drv_bits = "<br>".join(d for d in _drivers.get(_n["id"], []) if d)
            if _drv_bits:
                _drv_bits = "<br>" + _drv_bits
            _fig_dag.add_trace(go.Scatter(
                x=[_n["x"]], y=[_n["y"]],
                mode="markers+text",
                marker=dict(size=46, color=_ncol, opacity=0.92,
                            line=dict(color="white", width=2.5)),
                text=[_n["id"].replace("_", "<br>")],
                textposition="bottom center",
                textfont=dict(size=9, color=TEXT),
                showlegend=False,
                hovertemplate=(
                    f"<b>{_n['id'].replace('_',' ').title()}</b><br>Role: {_n['role']}"
                    f"{_val_bits}{_drv_bits}<extra></extra>"
                ),
            ))
        _pad = 1.6  # data-unit padding so edge-row labels don't clip against the axes
        _xs_all = [n["x"] for n in _dag_nodes]
        _ys_all = [n["y"] for n in _dag_nodes]
        _dagl = dict(**PLOTLY_LAYOUT)
        _dagl.update(dict(
            # The card this chart sits in is stretched (via CSS) to match
            # its taller neighbor card in the same row. A fixed small chart
            # height left that extra space as a dead blank strip below the
            # graph instead of the graph using it, so this is deliberately
            # tall enough to fill a typically-stretched card rather than
            # the plot's own minimum content size.
            height=620, plot_bgcolor="#FAFBFC",
            margin=dict(l=10, r=10, t=20, b=70),
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False,
                   "range": [min(_xs_all) - _pad, max(_xs_all) + _pad]},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False,
                   "range": [min(_ys_all) - _pad, max(_ys_all) + _pad]},
        ))
        _fig_dag.update_layout(**_dagl)

        _leg_html = '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px;">'
        for _role, _col in _role_colors.items():
            _leg_html += (
                f'<div style="display:flex;align-items:center;gap:4px;">'
                f'<span style="width:11px;height:11px;border-radius:50%;background:{_col};'
                f'display:inline-block;"></span>'
                f'<span style="font-size:0.75rem;color:{MUTED};">{_role}</span></div>'
            )
        _leg_html += (
            f'<div style="display:flex;align-items:center;gap:5px;">'
            f'<span style="width:26px;height:5px;border-radius:3px;'
            f'background:linear-gradient(90deg,#D7DEE8,{PRIMARY});display:inline-block;"></span>'
            f'<span style="font-size:0.75rem;color:{MUTED};">Edge thickness/color = pull strength '
            f'from your current lever settings</span></div>'
            '</div>'
        )
        st.markdown(_leg_html, unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.72rem;color:{SUBTLE};margin-bottom:4px;">'
            f'Hover any node or edge for the driving lever(s) and predicted value change.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_fig_dag, use_container_width=True, theme=None,
                        config={"displayModeBar": False})

# SECTION 4: Recovery Visualization (Horizontal Chart)
if not coefs.empty:
    chart_rows = []
    for _, row in coefs.iterrows():
        if pd.notna(row.get("ground_truth_value")) and pd.notna(row.get("estimated_value")):
            chart_rows.append(row)
            
    if chart_rows:
        st.markdown("<h4 style='color:#1E293B; margin-bottom:16px; margin-top:32px;'>Recovery Visualization</h4>", unsafe_allow_html=True)
        chart_df = pd.DataFrame(chart_rows)
        # Make sure we don't have too long labels
        edge_labels = (chart_df["parent"].str.replace("_", " ").str.title()
                       + " → "
                       + chart_df["child"].str.replace("_", " ").str.title())
        
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            name="Estimated", y=edge_labels, x=chart_df["estimated_value"],
            marker_color=PRIMARY, opacity=0.88, orientation='h'
        ))
        fig_coef.add_trace(go.Bar(
            name="Ground Truth", y=edge_labels, x=chart_df["ground_truth_value"],
            marker_color=WARNING, opacity=0.72, orientation='h'
        ))
        
        _cl = dict(**PLOTLY_LAYOUT)
        _cl.update(dict(
            barmode="group",
            title="Estimated vs Planted",
            height=max(300, len(chart_df) * 50), # dynamic height
            margin=dict(l=150), # more room for y-axis labels
            yaxis={"title": "Causal Edge", "autorange": "reversed"}, # reverse so first edge is at top
            xaxis={"title": "Coefficient Value"},
        ))
        fig_coef.update_layout(**_cl)
        try:
            st.plotly_chart(fig_coef, use_container_width=True, theme=None, config={'displayModeBar': False})
        except Exception as _e:
            st.error(f"Chart error: {_e}")

# SECTION 5: Technical Validation (Progressive Disclosure)
if not coefs.empty:
    with st.expander("Diagnostic Details"):
        st.markdown("Detailed coefficient recovery metrics and model fit quality.")
        
        tbl_rows = ""
        for _, row in coefs.iterrows():
            edge  = f"{row['parent']} → {row['child']}"
            est_v = row["estimated_value"]
            gt_v  = row.get("ground_truth_value", np.nan)
            ae_v  = row.get("abs_error", np.nan)
            pct_v = row.get("pct_error", np.nan)
            est_str = f"{est_v:+.4f}" if pd.notna(est_v) else "—"
            gt_str  = f"{gt_v:+.4f}" if pd.notna(gt_v)  else "—"
            ae_str  = f"{ae_v:.4f}"  if pd.notna(ae_v)  else "—"
            pct_str = f"{pct_v:.1%}" if pd.notna(pct_v) else "—"
            stat    = status_badge_html(row.get("status", "No Comparison"))
            mt      = row["model_type"]
            mdl_cls = {"logistic": "b-blue", "gradient_boosting": "b-amber"}.get(mt, "b-ok")
            mdl_str = f'<span class="badge {mdl_cls}">{mt}</span>'
            fit_str = f"{row.get('metric_label','R²')}={row.get('metric_value', 0):.3f}"
            row_sty = 'style="background:rgba(239,68,68,0.04);"' if row.get("status") == "Sign Error" else ""
            tbl_rows += (
                f"<tr {row_sty}>"
                f'<td class="ctbl-mono">{edge}</td>'
                f'<td class="ctbl-mono">{est_str}</td>'
                f'<td class="ctbl-mono">{gt_str}</td>'
                f'<td class="ctbl-mono">{ae_str}</td>'
                f'<td class="ctbl-mono" style="color:{MUTED};">{pct_str}</td>'
                f"<td>{stat}</td><td>{mdl_str}</td>"
                f'<td class="ctbl-mono" style="color:{MUTED};">{fit_str}</td></tr>'
            )

        st.markdown(
            f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch; margin-bottom: 24px;">'
            f'<table class="ctbl"><thead><tr>'
            f"<th>Edge (Parent → Child)</th><th>Effect Estimate</th><th>Validation Benchmark</th>"
            f"<th>Abs Error</th><th>% Error</th><th>Status</th><th>Model</th><th>Model Performance</th>"
            f"</tr></thead><tbody>{tbl_rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )
        
        eq_rows = [
            {"Node": node, "Model Type": eq["model_type"],
             "Metric": eq["metric_label"], "Value": f"{eq['r2_score']:.4f}"}
            for node, eq in scm.items()
        ]
        if eq_rows:
            st.markdown("#### Model Performance")
            render_table(pd.DataFrame(eq_rows))


# SECTION 5.5: Sensitivity to Unmeasured Confounding ─────────────────────────
# Already computed as part of the cached Double ML estimate (do_result) —
# zero extra cost. Runs DoWhy refutation tests: a placebo-treatment check,
# a random-common-cause check, and a sweep of estimates under increasing
# assumed unmeasured-confounding strength (5%-30%). Answers "how much would
# an unmeasured confounder have to matter before this conclusion flips?"
# rather than just reporting a single point estimate as if it were beyond
# doubt.
#
# Factored into a function (rather than left inline) because Case
# Inspection also renders this — every case's SHAP attribution rests on
# this same Double ML effect, so "how much should I trust the causal
# story behind this case" belongs there too. Defined here since this file
# execs first (see dashboard.py's tab loop) and shares it via globals().
def _render_sensitivity_section(expanded=False):
    if is_custom or stage_status.get("do_operator") != "ok" or not do_result:
        return
    _sens = do_result.get("sensitivity")
    if not _sens:
        return
    with st.expander("🛡️ Sensitivity to Unmeasured Confounding", expanded=expanded):
        st.markdown(
            "Every causal estimate here assumes the discovered DAG captures the "
            "relevant confounders. This section stress-tests that assumption "
            "directly, instead of asking you to take it on faith."
        )

        _placebo_ok = _sens.get("placebo_passes", True)
        _rc_ok      = _sens.get("random_cause_stable", True)
        _e_val      = _sens.get("e_value")

        _b1, _b2, _b3 = st.columns(3)
        with _b1:
            st.metric(
                "Placebo Test",
                "Pass" if _placebo_ok else "Fail",
                help="Permutes the treatment randomly and re-estimates. A real "
                     "causal effect should vanish (~0) under a fake, permuted "
                     "treatment; if it doesn't, the model may be picking up "
                     "spurious structure rather than a true causal path.",
            )
            st.caption(f"Effect under permuted treatment: {_sens.get('placebo_effect', 0):+.3f} days (expect ~0)")
        with _b2:
            st.metric(
                "Random Common Cause",
                "Stable" if _rc_ok else "Unstable",
                help="Adds a random, independent variable as a fake confounder "
                     "and re-estimates. A robust estimate shouldn't move much "
                     "when an irrelevant variable is added to the adjustment set.",
            )
            st.caption(f"Re-estimate with noise variable: {_sens.get('random_cause_estimate', 0):+.3f} days")
        with _b3:
            st.metric(
                "E-value",
                f"{_e_val:.1f}" if _e_val is not None else "—",
                help="Minimum strength an unmeasured confounder would need "
                     "(on the risk-ratio scale) to fully explain away the "
                     "estimated effect. Higher = more robust.",
            )
            st.caption("Confounder strength needed to nullify the result")

        _strengths = _sens.get("confounding_strengths", [])
        _est_range = _sens.get("estimates_under_confounding", [])
        if _strengths and _est_range:
            _fig_sens = go.Figure()
            _fig_sens.add_trace(go.Scatter(
                x=[f"{s:.0%}" for s in _strengths], y=_est_range,
                mode="lines+markers", name="Estimate under assumed confounding",
                line=dict(color=WARNING, width=2.5), marker=dict(size=8),
            ))
            _fig_sens.add_hline(
                y=do_result.get("causal", 0), line_dash="dash", line_color=PRIMARY,
                annotation_text=f"Reported estimate: {do_result.get('causal', 0):+.2f} days",
                annotation_font_color=PRIMARY,
            )
            _sens_layout = dict(**PLOTLY_LAYOUT)
            _sens_layout.update(dict(
                title="Causal Estimate vs. Assumed Unmeasured Confounder Strength",
                height=320, margin=dict(l=20, r=20, t=40, b=40),
                xaxis={"title": "Assumed confounder strength (effect on treatment & outcome)"},
                yaxis={"title": f"Estimated effect ({cfg.get('outcome_label','days')})"},
                showlegend=False,
            ))
            _fig_sens.update_layout(**_sens_layout)
            try:
                st.plotly_chart(_fig_sens, use_container_width=True, theme=None,
                                config={'displayModeBar': False})
            except Exception as _e:
                st.error(f"Chart error: {_e}")

        st.caption(_sens.get("verdict", ""))

_render_sensitivity_section()


# ── SECTION 2.5: Treatment Effect Heterogeneity (CATE) ──────────────────────
if not is_custom:
    _mod_var   = cfg.get("moderator_var", "")
    _mod_label = cfg.get("moderator_label", "Complexity")
    _treat_lbl = cfg["treatment_var"].replace("_", " ").title()
    _out_lbl   = cfg["outcome_label"]

    if _mod_var and _mod_var in df.columns:
        with st.spinner("Computing treatment effect heterogeneity…"):
            # Cached on (domain, n, seed) — this is ~30 Double-ML-with-GBM
            # fits (~13s); previously recomputed on every rerun (including
            # tab switches) since it was called uncached, directly.
            _cate_data = _compute_cate(domain, n_int, seed_int)

        if _cate_data:
            st.markdown(
                "<h4 style='color:#1E293B; margin-bottom:4px; margin-top:32px;'>"
                "Treatment Effect Heterogeneity</h4>"
                f"<p style='color:#64748B; font-size:0.88rem; margin-bottom:16px;'>"
                f"Does the causal effect of <b>{_treat_lbl}</b> on <b>{_out_lbl}</b> "
                f"differ across <b>{_mod_label}</b> segments? "
                f"Each bar is a Double ML estimate within that subgroup. "
                f"Error bars show 95% CI.</p>",
                unsafe_allow_html=True,
            )

            _cate_labels   = [r["label"]    for r in _cate_data]
            _cate_ests     = [r["estimate"] for r in _cate_data]
            _cate_ci_lo    = [r["ci_low"]   for r in _cate_data]
            _cate_ci_hi    = [r["ci_high"]  for r in _cate_data]
            _cate_ns       = [r["n"]        for r in _cate_data]
            _cate_err_lo   = [e - l for e, l in zip(_cate_ests, _cate_ci_lo)]
            _cate_err_hi   = [h - e for e, h in zip(_cate_ests, _cate_ci_hi)]

            # Segment colours: Low=blue, Mid=amber, High=red
            _seg_colors = ["#3B82F6", "#F59E0B", "#EF4444"][:len(_cate_data)]

            _fig_cate = go.Figure()
            _fig_cate.add_trace(go.Bar(
                x=_cate_labels,
                y=_cate_ests,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=_cate_err_hi,
                    arrayminus=_cate_err_lo,
                    color="#94A3B8",
                    thickness=2,
                    width=6,
                ),
                marker_color=_seg_colors,
                marker_line=dict(color="white", width=1.5),
                opacity=0.88,
                text=[f"{e:+.2f}" for e in _cate_ests],
                textposition="outside",
                textfont=dict(size=13, color="#1E293B"),
                customdata=_cate_ns,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "CATE: %{y:+.3f} days<br>"
                    "95% CI: [%{error_y.arrayminus:.3f} – %{error_y.array:.3f}]<br>"
                    "n = %{customdata}<extra></extra>"
                ),
                name="CATE",
                showlegend=False,
            ))

            # Average treatment effect reference line
            _avg_est = sum(_cate_ests) / len(_cate_ests)
            _fig_cate.add_hline(
                y=_avg_est,
                line_dash="dot",
                line_color="#6366F1",
                line_width=1.5,
                annotation_text=f"  ATE ≈ {_avg_est:+.2f} (binary full-switch vs no treatment)",
                annotation_font=dict(color="#6366F1", size=11),
                annotation_position="right",
            )

            _cate_layout = dict(**PLOTLY_LAYOUT)
            _cate_layout.update(dict(
                height=340,
                margin=dict(l=20, r=80, t=30, b=60),
                yaxis={**PLOTLY_LAYOUT.get("yaxis", {}),
                       "title": f"Causal Effect on {_out_lbl}",
                       "title_font": dict(size=12),
                       "zeroline": True,
                       "zerolinecolor": "#E2E8F0",
                       "zerolinewidth": 1.5},
                xaxis={**PLOTLY_LAYOUT.get("xaxis", {}),
                       "title": _mod_label,
                       "title_font": dict(size=12)},
            ))
            _fig_cate.update_layout(**_cate_layout)
            st.plotly_chart(_fig_cate, use_container_width=True, theme=None,
                            config={"displayModeBar": False})

            # Insight card
            _max_cate = max(_cate_data, key=lambda r: r["estimate"])
            _min_cate = min(_cate_data, key=lambda r: r["estimate"])
            _spread   = _max_cate["estimate"] - _min_cate["estimate"]
            _spread_pct = (_spread / abs(_min_cate["estimate"]) * 100
                           if abs(_min_cate["estimate"]) > 0.01 else 0)
            st.markdown(
                f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; '
                f'border-left:4px solid #6366F1; border-radius:8px; '
                f'padding:14px 18px; margin-top:4px; margin-bottom:8px;">'
                f'<div style="color:#6366F1; font-size:0.72rem; font-weight:800; '
                f'text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Conditional Average Treatment Effect (CATE)</div>'
                f'<div style="color:#1E293B; font-size:0.95rem; line-height:1.6;">'
                f'The causal effect of <b>{_treat_lbl}</b> is strongest in the '
                f'<b>{_max_cate["label"].split(" (")[0]}</b> complexity segment '
                f'(<b>{_max_cate["estimate"]:+.2f} days</b>), '
                f'{_spread_pct:.0f}% larger than the '
                f'{_min_cate["label"].split(" (")[0]} segment '
                f'({_min_cate["estimate"]:+.2f} days). '
                f'This heterogeneity suggests <b>targeted interventions</b> '
                f'would yield different returns by {_mod_label.lower()} profile.'
                f'</div></div>',
                unsafe_allow_html=True,
            )

# Cross-domain validation note — condensed to a static summary (the live n=500
# benchmark recompute + scorecards were removed as an unnecessary extra tab;
# see project history for the full Manufacturing-vs-Healthcare benchmark).
# Closing card for this tab (rendered last, after CATE) — kept as a named
# variable, not inlined, because tab_decision_intelligence.py also reuses
# this exact global (via the shared exec globals() dict) inside its own
# "Cross-domain validation benchmark" expander.
_domain_validation_note_html = (
    '<div style="background:#F8FAFC; border:1px solid #CBD5E1; padding:24px; border-radius:12px;">'
    '<div style="text-align:center;">'
    '<h4 style="color:#1E293B; font-weight:900; letter-spacing:-0.5px; margin-bottom:8px;">DOMAIN-AGNOSTIC CAUSAL INTELLIGENCE</h4>'
    '<div style="color:#94A3B8;font-size:0.78rem;font-weight:600;margin-bottom:16px;">Controlled benchmark · n=500 synthetic cases per domain · Pre-domain-knowledge discovery</div>'
    '<p style="color:#334155; font-size:1rem; line-height:1.6; max-width:800px; margin:0 auto 20px auto;">'
    'CausalOCPM successfully recovered confounding structures and true causal effects across Manufacturing and Healthcare without requiring domain-specific modifications.</p>'
    '<div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">'
    '<div style="background:#F0FDF4; color:#059669; padding:6px 14px; border-radius:24px; font-weight:700; font-size:0.85rem; border:1px solid #BBF7D0;">✓ Generalises Across Industries</div>'
    '<div style="background:#F0FDF4; color:#059669; padding:6px 14px; border-radius:24px; font-weight:700; font-size:0.85rem; border:1px solid #BBF7D0;">✓ Preserves Causal Validity</div>'
    '<div style="background:#F0FDF4; color:#059669; padding:6px 14px; border-radius:24px; font-weight:700; font-size:0.85rem; border:1px solid #BBF7D0;">✓ Requires No Custom Redesign</div>'
    '</div>'
    '</div>'
    '</div>'
)
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
st.markdown(_domain_validation_note_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — CEO DECISION INTELLIGENCE REPORT
# ══════════════════════════════════════════════════════════════════════════════
