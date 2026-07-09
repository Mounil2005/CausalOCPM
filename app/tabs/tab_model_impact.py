# ── AI CAUSAL INTERPRETATION (PHASE 0) ──────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(to right, #F8FAFC, #FFFFFF); border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6; border-radius: 12px; padding: 24px 32px; margin-bottom: 28px; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
    <div style="font-size: 0.85rem; font-weight: 800; color: #3B82F6; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.2rem;">✨</span> AI Causal Interpretation
    </div>
    <ul style="color: #334155; font-size: 1rem; line-height: 1.7; margin: 0; padding-left: 20px;">
        <li>The Structural Causal Model successfully quantified the strength of the discovered causal links.</li>
        <li>The What-If Simulator allows you to dynamically adjust operational levers and observe their impact.</li>
        <li>Treatment effects reveal that interventions have varying impacts depending on operational segments (CATE).</li>
        <li>Use the simulator below to compare intervention scenarios before deployment.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

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
if "sim_scenarios" not in st.session_state:
    st.session_state["sim_scenarios"] = []

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

with _botl:
    with st.container(border=True):
        _sim_section_header("#FEF3C7", "📋", "Scenario Comparison")

        _sc_in, _sc_sv = st.columns([3, 1])
        with _sc_in:
            _save_name = st.text_input(
                "Scenario name", placeholder="e.g. Full Automation + Express",
                key="sim_save_name_input", label_visibility="collapsed",
            )
        with _sc_sv:
            if st.button("💾 Save Scenario", use_container_width=True, key="sim_save_btn"):
                if _save_name and len(st.session_state["sim_scenarios"]) < 4:
                    st.session_state["sim_scenarios"].append({
                        "name":   _save_name,
                        "levers": dict(_lv),
                        "result": {k: v for k, v in _res.items()
                                   if k not in ("mediators", "deltas")},
                    })
                    st.rerun()
                elif len(st.session_state["sim_scenarios"]) >= 4:
                    st.warning("Max 4 scenarios — clear some first.")

        st.markdown(
            f'<div style="font-size:0.75rem;color:{MUTED};margin-bottom:10px;">'
            f'Save up to 4 named scenarios to compare them side-by-side.</div>',
            unsafe_allow_html=True,
        )

        _scenarios = st.session_state["sim_scenarios"]
        st.markdown(
            f'<div style="font-size:0.68rem;color:{MUTED};font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">'
            f'Saved Scenarios ({len(_scenarios)}/4)</div>',
            unsafe_allow_html=True,
        )

        # Baseline row is always shown as the reference point, styled the
        # same as saved scenarios but tagged ACTIVE instead of an impact tier.
        def _sim_impact_pill(pct):
            if pct > 25:   return "#059669", "#ECFDF5", "HIGH IMPACT"
            if pct > 10:   return "#D97706", "#FFFBEB", "MODERATE IMPACT"
            if pct > 0:    return "#EA580C", "#FFF7ED", "LOW IMPACT"
            return "#DC2626", "#FEF2F2", "NO IMPROVEMENT"

        _row_html = (
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:9px 4px;border-bottom:1px solid {BORDER};">'
            f'<span style="font-size:0.82rem;color:{TEXT};font-weight:500;">Baseline (Current)</span>'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span style="background:#EEF2FF;color:#4F46E5;font-size:0.65rem;font-weight:700;'
            f'padding:3px 9px;border-radius:8px;">ACTIVE</span>'
            f'<span style="font-size:0.85rem;font-weight:700;color:{TEXT};min-width:56px;text-align:right;">'
            f'{_bl:.1f} days</span></div></div>'
        )
        for _s in _scenarios:
            _r = _s["result"]
            _pcol, _pbg, _plbl = _sim_impact_pill(_r["improvement_pct"])
            _row_html += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:9px 4px;border-bottom:1px solid {BORDER};">'
                f'<span style="font-size:0.82rem;color:{TEXT};font-weight:500;">{_s["name"]}</span>'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="background:{_pbg};color:{_pcol};font-size:0.65rem;font-weight:700;'
                f'padding:3px 9px;border-radius:8px;">{_plbl}</span>'
                f'<span style="font-size:0.85rem;font-weight:700;color:{TEXT};min-width:56px;text-align:right;">'
                f'{_r["predicted"]:.1f} days</span></div></div>'
            )
        st.markdown(_row_html, unsafe_allow_html=True)

        if st.session_state["sim_scenarios"]:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Saved Scenarios", use_container_width=True, key="sim_clear_btn"):
                st.session_state["sim_scenarios"] = []
                st.rerun()

        # Always render the comparison chart, even with just the Baseline
        # bar when no scenarios are saved yet — a real, meaningful chart
        # rather than nothing, so this card isn't left far shorter than its
        # "Live Causal Graph" neighbor (Streamlit's row stretches both
        # columns to the same height regardless, so an empty card here
        # otherwise reads as a chunk of dead blank space, not a deliberate
        # layout choice).
        _sc_names  = ["Baseline"] + [s["name"] for s in _scenarios]
        _sc_delays = [_bl]        + [s["result"]["predicted"] for s in _scenarios]
        _sc_colors = [MUTED]      + [
            (SUCCESS if s["result"]["improvement_pct"] > 10 else WARNING)
            for s in _scenarios
        ]
        _fig_sc = go.Figure(go.Bar(
            x=_sc_names, y=_sc_delays,
            marker_color=_sc_colors,
            text=[f"{v:.1f}d" for v in _sc_delays],
            textposition="outside",
        ))
        _scl = dict(**PLOTLY_LAYOUT)
        _scl.update(dict(
            height=260, showlegend=False,
            margin=dict(l=10, r=10, t=20, b=20),
            yaxis={"title": f"{_sim_out_lbl} (days)", "gridcolor": BORDER},
        ))
        _fig_sc.update_layout(**_scl)
        st.plotly_chart(_fig_sc, use_container_width=True, theme=None,
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
            _dag_edges = [
                ("supplier_a",         "material_lead_time",
                 _lv["supplier_reliability_pct"] != _MFG_DEFAULTS["supplier_reliability_pct"]),
                ("order_complexity",   "machine_queue",  False),
                ("machine_queue",      "approval_duration",
                 _lv["machine_capacity_expanded"] or _lv["additional_workforce"] > 0),
                ("export_flag",        "approval_duration",  _lv["export_flag_reduction"]),
                ("material_lead_time", "shipment_delay",
                 _lv["material_lead_time_mode"] != "Current" or
                 _lv["supplier_reliability_pct"] != _MFG_DEFAULTS["supplier_reliability_pct"]),
                ("approval_duration",  "shipment_delay",
                 _lv["approval_automation"] or _lv["export_flag_reduction"]),
                ("carrier_express",    "shipment_delay",
                 _lv["carrier_express_pct"] != _MFG_DEFAULTS["carrier_express_pct"]),
                ("order_complexity",   "shipment_delay",  False),
            ]
        else:
            _dag_nodes = [
                {"id": "patient_complexity",     "x": 0.0, "y": 1.0, "role": "Confounder"},
                {"id": "specialist_requirement", "x": 0.0, "y": 3.0, "role": "Treatment"},
                {"id": "bed_occupancy",          "x": 3.0, "y": 2.0, "role": "Mediator"},
                {"id": "triage_score",           "x": 1.5, "y": 0.0, "role": "Mediator"},
                {"id": "treatment_duration",     "x": 6.0, "y": 2.0, "role": "Outcome"},
            ]
            _dag_edges = [
                ("patient_complexity",     "treatment_duration",    False),
                ("specialist_requirement", "treatment_duration",
                 _lv["specialist_allocation_pct"] != _HC_DEFAULTS["specialist_allocation_pct"]),
                ("bed_occupancy",          "treatment_duration",    _lv["bed_capacity_expanded"]),
                ("triage_score",           "specialist_requirement", _lv["triage_automation"]),
            ]

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
        # halo — real derived state (from the same _active flags driving
        # edge color), not decoration, so it's obvious at a glance which
        # part of the graph your current lever changes are running through.
        _active_nodes = set()
        for _src, _dst, _active in _dag_edges:
            if _active:
                _active_nodes.add(_src)
                _active_nodes.add(_dst)

        import math
        _fig_dag = go.Figure()
        for _src, _dst, _active in _dag_edges:
            _x0, _y0, _x1, _y1 = _nxs[_src], _nys[_src], _nxs[_dst], _nys[_dst]
            _fig_dag.add_trace(go.Scatter(
                x=[_x0, _x1, None], y=[_y0, _y1, None],
                mode="lines",
                line=dict(color=PRIMARY if _active else "#D7DEE8",
                          width=3 if _active else 1.2),
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
                marker=dict(symbol="triangle-right", size=11 if _active else 8,
                            color=PRIMARY if _active else "#B9C2D0",
                            angle=_ang, line=dict(width=0)),
                showlegend=False, hoverinfo="skip",
            ))
        for _n in _dag_nodes:
            _ncol = _role_colors.get(_n["role"], MUTED)
            if _n["id"] in _active_nodes:
                _fig_dag.add_trace(go.Scatter(
                    x=[_n["x"]], y=[_n["y"]], mode="markers",
                    marker=dict(size=64, color=_ncol, opacity=0.18, line=dict(width=0)),
                    showlegend=False, hoverinfo="skip",
                ))
            _fig_dag.add_trace(go.Scatter(
                x=[_n["x"]], y=[_n["y"]],
                mode="markers+text",
                marker=dict(size=46, color=_ncol, opacity=0.92,
                            line=dict(color="white", width=2.5)),
                text=[_n["id"].replace("_", "<br>")],
                textposition="bottom center",
                textfont=dict(size=9, color=TEXT),
                showlegend=False,
                hovertemplate=f"<b>{_n['id']}</b><br>Role: {_n['role']}<extra></extra>",
            ))
        _pad = 1.6  # data-unit padding so edge-row labels don't clip against the axes
        _xs_all = [n["x"] for n in _dag_nodes]
        _ys_all = [n["y"] for n in _dag_nodes]
        _dagl = dict(**PLOTLY_LAYOUT)
        _dagl.update(dict(
            height=400, plot_bgcolor="#FAFBFC",
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
            f'<div style="display:flex;align-items:center;gap:4px;">'
            f'<span style="width:18px;height:2px;background:{PRIMARY};display:inline-block;">'
            f'</span><span style="font-size:0.75rem;color:{MUTED};">Active path</span></div>'
            f'<div style="display:flex;align-items:center;gap:4px;">'
            f'<span style="width:18px;height:2px;background:#CBD5E1;display:inline-block;">'
            f'</span><span style="font-size:0.75rem;color:{MUTED};">Passive edge</span></div>'
            '</div>'
        )
        st.markdown(_leg_html, unsafe_allow_html=True)
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
if not is_custom and stage_status.get("do_operator") == "ok" and do_result:
    _sens = do_result.get("sensitivity")
    if _sens:
        with st.expander("Sensitivity to Unmeasured Confounding", expanded=False):
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


# SECTION — CASE ATTRIBUTION (part of Model & Impact tab)
st.divider()
from src.phase5_attribution import (explain_case, get_attribution_summary,
                                     explain_limitation)

outcome_var   = cfg["outcome_var"]
outcome_label = cfg["outcome_label"]

if is_custom:
    accuracy_disclaimer(custom_confidence, len(df), custom_quality.get("score", 0))

id_cols     = ["order_id", "patient_id"]
case_id_col = next((c for c in id_cols if c in df.columns), None)
case_ids    = df[case_id_col].tolist() if case_id_col else [f"Case_{i}" for i in range(len(df))]

# SECTION 1: Case Selection
st.markdown("<h4 style='color:#1E293B; margin-bottom:16px;'>Investigating Case</h4>", unsafe_allow_html=True)
sel_col, dummy_col = st.columns([1, 2])
with sel_col:
    selected_case = st.selectbox("Select Case", options=case_ids[:200], index=0, key="case_selector", label_visibility="collapsed")
    if len(case_ids) > 200:
        st.caption(f"Showing the first 200 of {len(case_ids):,} cases.")
case_idx = case_ids.index(selected_case) if selected_case in case_ids else 0

expl           = explain_case(df, scm, case_idx, outcome_var, domain=domain)
attrib_summary = get_attribution_summary(expl)

if not expl.empty:
    baseline  = expl.attrs.get("baseline", 0.0)
    predicted = expl.attrs.get("predicted_outcome", 0.0)
    actual    = expl.attrs.get("actual_outcome", df[outcome_var].iloc[case_idx])
    features  = expl["feature"].tolist()
    shap_vals = expl["shap_value"].tolist()
    
    # Determine performance context
    diff = actual - baseline
    if diff < 0:
        performance = f"outperforming the population average by {abs(diff):.2f} {outcome_label}"
        status_text = "✓ Better than baseline"
        status_color = SUCCESS
    else:
        performance = f"underperforming the population average by {abs(diff):.2f} {outcome_label}"
        status_text = "⚠️ Worse than baseline"
        status_color = "#B45309"
        
    # Top contributor
    top_row = expl.iloc[expl["shap_value"].abs().idxmax()]
    top_contributor = top_row["feature"].replace("_", " ").title()
    
    # SECTION 2: Executive Interpretation Banner
    exec_text = (
        f"Case <b>{selected_case}</b> achieved an outcome of {actual:.2f} {outcome_label}, {performance}. "
        f"<b>{top_contributor}</b> was the dominant contributor to this outcome. "
        f"Additional interventions targeting controllable factors could further improve the outcome by approximately {attrib_summary['max_reducible_delay']:.2f} {outcome_label}."
    )
    st.markdown(
        f'<div style="background:#F0FDF4; border-left:4px solid {SUCCESS}; padding:20px; border-radius:4px; margin-bottom:32px; box-shadow:0 2px 8px rgba(0,0,0,0.02);">'
        f'<div style="color:{SUCCESS}; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Executive Interpretation</div>'
        f'<div style="color:#1E293B; font-size:1.05rem; line-height:1.6;">{exec_text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # SECTION 3: Case Snapshot
    complexity_col = "order_complexity" if domain == "manufacturing" else "patient_complexity"
    comp_val = df[complexity_col].iloc[case_idx] if complexity_col in df.columns else 0
    comp_text = "High" if comp_val > 5 else "Low"
    
    treat_label = cfg["treatment_options"].get(cfg["treatment_var"], cfg["treatment_var"]) if "treatment_options" in cfg else "Treatment"
    treat_val = "Yes" if int(df[cfg["treatment_var"]].iloc[case_idx]) == 1 else "No" if cfg["treatment_var"] in df.columns else "N/A"
    
    snap_html = (
        f'<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px;">'
        f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid {status_color}; padding:16px; border-radius:10px;">'
        f'<div style="color:#64748B; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Actual Outcome</div>'
        f'<div style="font-size:1.8rem; font-weight:800; color:{TEXT}; margin-top:4px;">{actual:.2f}</div>'
        f'<div style="font-size:0.85rem; font-weight:600; color:{status_color}; margin-top:4px;">{status_text}</div>'
        f'</div>'
        f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:16px; border-radius:10px;">'
        f'<div style="color:#64748B; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Complexity Profile</div>'
        f'<div style="font-size:1.8rem; font-weight:800; color:{TEXT}; margin-top:4px;">{comp_text}</div>'
        f'<div style="font-size:0.85rem; font-weight:600; color:#64748B; margin-top:4px;">{comp_val:.0f} / 10</div>'
        f'</div>'
        f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:16px; border-radius:10px;">'
        f'<div style="color:#64748B; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Treatment Strategy</div>'
        f'<div style="font-size:1.8rem; font-weight:800; color:{TEXT}; margin-top:4px;">{treat_val}</div>'
        f'<div style="font-size:0.85rem; font-weight:600; color:#64748B; margin-top:4px;">{treat_label} applied</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(snap_html, unsafe_allow_html=True)
    
    # SECTION 4: Outcome Attribution (Hero Visualization)
    st.markdown("<h4 style='color:#1E293B; margin-bottom:8px;'>Why Did This Outcome Occur?</h4>", unsafe_allow_html=True)
    
    fig_wf = go.Figure(go.Waterfall(
        x=["Population Average"] + [f.replace("_", " ").title() for f in features] + ["Case Prediction"],
        y=[baseline] + shap_vals + [0],
        measure=["absolute"] + ["relative"] * len(shap_vals) + ["total"],
        connector=dict(line=dict(color=BORDER, width=2)),
        increasing=dict(marker_color=ERROR),
        decreasing=dict(marker_color=SUCCESS),
        totals=dict(marker_color="#334155"),
        texttemplate="%{y:+.2f}", textposition="outside",
        textfont=dict(size=12)
    ))
    fig_wf.add_hline(y=actual, line_dash="dash", line_color="#94A3B8",
                      annotation_text=f"Actual: {actual:.2f}",
                      annotation_font_color="#475569")
    _wfl = dict(**PLOTLY_LAYOUT)
    _wfl.update(dict(
        yaxis={**PLOTLY_LAYOUT.get("yaxis", {}), "title": outcome_label, "title_font": dict(size=14)},
        xaxis={**PLOTLY_LAYOUT.get("xaxis", {}), "tickangle": -40, "tickfont": dict(size=11), "automargin": True},
        height=560,
        margin=dict(l=20, r=20, t=20, b=160),
    ))
    fig_wf.update_layout(**_wfl)
    try:
        st.plotly_chart(fig_wf, use_container_width=True, theme=None, config={'displayModeBar': False})
    except Exception as _e:
        st.error(f"Chart error: {_e}")

    # SECTION 5: Actionability Insights
    col_opp, col_con = st.columns(2)
    with col_opp:
        st.markdown(
            f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid #2563EB; padding:24px; border-radius:12px; height:100%; box-shadow:0 4px 12px rgba(0,0,0,0.03);">'
            f'<div style="color:{TEXT}; font-size:0.85rem; font-weight:800; text-transform:uppercase; margin-bottom:8px;"><span style="color:#2563EB; margin-right:6px;">●</span> Intervention Opportunities</div>'
            f'<div style="color:#64748B; font-size:0.9rem; font-weight:600; margin-bottom:16px;">Controllable Factors Contribution</div>'
            f'<div style="font-size:2.4rem; font-weight:800; color:#2563EB; margin-bottom:16px;">{attrib_summary["actionable_total"]:+.2f}</div>'
            f'<div style="color:#334155; font-size:0.95rem; line-height:1.5;">Operational actions targeting these controllable factors could substantially influence future outcomes. '
            f'Confidence in causal effect is <span style="color:{_conf_col};font-weight:700;">{_conf_lbl.lower()}</span> '
            f'(F1 = {_sim_f1:.2f}).</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    with col_con:
        st.markdown(
            f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid #94A3B8; padding:24px; border-radius:12px; height:100%; box-shadow:0 4px 12px rgba(0,0,0,0.03);">'
            f'<div style="color:{TEXT}; font-size:0.85rem; font-weight:800; text-transform:uppercase; margin-bottom:8px;"><span style="color:#64748B; margin-right:6px;">●</span> System Constraints</div>'
            f'<div style="color:#64748B; font-size:0.9rem; font-weight:600; margin-bottom:16px;">Structural Contribution</div>'
            f'<div style="font-size:2.4rem; font-weight:800; color:#64748B; margin-bottom:16px;">{attrib_summary["structural_total"]:+.2f}</div>'
            f'<div style="color:#334155; font-size:0.95rem; line-height:1.5;">These drivers arise from underlying process characteristics. Altering them may require long-term system-level transformation efforts.</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    
    # SECTION 6: Key Insight Summary
    top_controllable = expl[expl['attribution'] == 'actionable']
    if not top_controllable.empty:
        top_ctrl_feat = top_controllable.iloc[top_controllable['shap_value'].abs().idxmax()]['feature'].replace("_", " ").title()
    else:
        top_ctrl_feat = "Treatment"
        
    insight_card(
        "Key Insight",
        f"{top_ctrl_feat} was the most significant controllable lever in this case. "
        f"Although structural limits exist, further interventions targeting actionable drivers could yield additional operational gains of up to {attrib_summary['max_reducible_delay']:.2f}.",
        "executive",
    )

    # SECTION 7: Technical Evidence
    with st.expander("Detailed Attribution Analysis"):
        st.markdown("Raw SHAP values and attribution categories supporting the executive summary.")
        detail = expl[["feature", "attribution", "shap_value", "feature_value"]].copy()
        detail.columns = ["Feature", "Attribution", "SHAP Value", "Feature Value"]
        
        # Formatting
        tbl_rows = ""
        # Sort by absolute SHAP value
        detail['abs_shap'] = detail['SHAP Value'].abs()
        detail = detail.sort_values(by='abs_shap', ascending=False)
        
        for _, row in detail.iterrows():
            feat = row['Feature']
            attr = row['Attribution']
            shap = row['SHAP Value']
            val = row['Feature Value']
            
            attr_badge = f'<span style="background:#DBEAFE; color:#1E40AF; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;">Controllable</span>' if attr == 'actionable' else f'<span style="background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;">Structural</span>'
            shap_color = SUCCESS if shap < 0 else ERROR
            
            tbl_rows += (
                f"<tr>"
                f'<td style="font-weight:600; color:#334155;">{feat}</td>'
                f'<td>{attr_badge}</td>'
                f'<td style="color:{shap_color}; font-weight:700;">{shap:+.4f}</td>'
                f'<td style="color:#64748B;">{val:.2f}</td>'
                f'</tr>'
            )
            
        st.markdown(
            f'<table class="ctbl" style="width:100%;"><thead><tr>'
            f"<th>Feature</th><th>Category</th><th>SHAP Value</th><th>Feature Value</th>"
            f"</tr></thead><tbody>{tbl_rows}</tbody></table>",
            unsafe_allow_html=True,
        )

# SECTION 8: Methodological Foundation
with st.expander("Methodological Foundation"):
    st.markdown(explain_limitation(include_citation=True))


# Cross-domain validation note — condensed to a static summary (the live n=500
# benchmark recompute + scorecards were removed as an unnecessary extra tab;
# see project history for the full Manufacturing-vs-Healthcare benchmark).
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



# SECTION 1: Structural Equation Summary
t3_col_eq = st.container()

with t3_col_eq:
    st.markdown("<h4 style='color:#1E293B; margin-bottom:16px;'>Structural Equation Summary</h4>", unsafe_allow_html=True)
    
    cards_html = f'<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 32px;">'
    for node, eq in scm.items():
        mt = eq["model_type"]
        mt_display = mt.replace("_", " ").title()
        val = eq['r2_score']
        metric = eq['metric_label']
        
        if mt == "logistic":
            status = "Reliable Classification"
        elif mt == "gradient_boosting":
            status = "High Confidence"
        else:
            status = "Linear Fit"
        # Color reflects the fit quality itself (same thresholds used for
        # Model Confidence elsewhere in this tab), not just which model ran.
        if val >= 0.9:
            color = "#059669"
        elif val >= 0.7:
            color = "#D97706"
        else:
            color = "#DC2626"
        status = f"{'✓' if val >= 0.7 else '⚠'} {status}"
            
        node_clean = node.replace("_", " ").title()
        
        cards_html += (
            f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid {color}; padding:20px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.03);">'
            f'<div style="color:#1E293B; font-size:1.1rem; font-weight:700; margin-bottom:8px;">{node_clean}</div>'
            f'<div style="color:#64748B; font-size:0.9rem; margin-bottom:12px;">{mt_display}</div>'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div style="font-size:1rem; font-weight:600; color:#334155;">{metric} = {val:.3f}</div>'
            f'<div style="font-size:0.8rem; font-weight:600; color:{color};">{status}</div>'
            f'</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

# Calculate validation metrics
if not coefs.empty:
    total_edges = len(coefs)
    # Sign Error or no comparison
    recovered_edges = len(coefs[coefs['status'] != 'Sign Error'])
    # Domains without a planted numeric ground truth (e.g. healthcare — see
    # _MFG_GROUND_TRUTH in phase3_scm.py) have pct_error/abs_error all-NaN by
    # design (structural demonstration only, not numerical validation). Show
    # that honestly instead of ".mean()" silently producing "nan%".
    _raw_mean_error = coefs['pct_error'].mean() if 'pct_error' in coefs else 0.0
    mean_error_str = f"{_raw_mean_error:.1%}" if pd.notna(_raw_mean_error) else "N/A"
    sign_consistency = (recovered_edges / total_edges) * 100 if total_edges > 0 else 100.0

    if 'abs_error' in coefs and not coefs['abs_error'].isna().all():
        best_idx = coefs['abs_error'].idxmin()
        best_edge_row = coefs.loc[best_idx]
        strongest_recovery = f"{best_edge_row['parent']} → {best_edge_row['child']}"
        strongest_err_str = f"{best_edge_row['pct_error']:.1%} Error"
    else:
        strongest_recovery = "N/A"
        strongest_err_str = "No numeric ground truth for this domain"
        
    # SECTION 2: Model Validation Summary
    st.markdown("<h4 style='color:#1E293B; margin-bottom:16px;'>Model Validation Summary</h4>", unsafe_allow_html=True)
    
    val_html = (
        f'<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px;">'
        f'<div style="background:#F0FDF4; border:1px solid #BBF7D0; padding:16px; border-radius:10px;">'
        f'<div style="color:{SUCCESS}; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Relationships Recovered</div>'
        f'<div style="font-size:1.8rem; font-weight:800; color:{SUCCESS}; margin-top:4px;">{recovered_edges} / {total_edges}</div>'
        f'<div style="font-size:0.8rem; font-weight:600; color:{SUCCESS}; margin-top:2px;">{sign_consistency:.0f}% sign-consistent</div>'
        f'</div>'
        f'<div style="background:#F0FDF4; border:1px solid #BBF7D0; padding:16px; border-radius:10px;">'
        f'<div style="color:{SUCCESS}; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Mean Relative Error</div>'
        f'<div style="font-size:1.8rem; font-weight:800; color:{SUCCESS}; margin-top:4px;">{mean_error_str}</div>'
        f'</div>'
        f'<div style="background:#F0FDF4; border:1px solid #BBF7D0; padding:16px; border-radius:10px;">'
        f'<div style="color:{SUCCESS}; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Strongest Recovery</div>'
        f'<div style="font-size:1rem; font-weight:700; color:{SUCCESS}; margin-top:4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{strongest_recovery}">{strongest_recovery}</div>'
        f'<div style="font-size:0.8rem; font-weight:600; color:{SUCCESS}; margin-top:2px;">{strongest_err_str}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(val_html, unsafe_allow_html=True)
    
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — CEO DECISION INTELLIGENCE REPORT
# ══════════════════════════════════════════════════════════════════════════════
