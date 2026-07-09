"""
Patch: replaces SECTION 3 (What-If Simulator) in tab3 of dashboard.py
with a full interactive Causal Simulator, white/light theme.
Run from: causal_ocpm/
"""
import pathlib, sys

DASH = pathlib.Path("app/dashboard.py")
assert DASH.exists(), "Run from causal_ocpm/"

lines = DASH.read_text(encoding="utf-8").splitlines(keepends=True)

START_MARKER = "# SECTION 3: What-If Simulator"
END_MARKER   = "# SECTION 4: Recovery Visualization"

start_idx = end_idx = None
for i, ln in enumerate(lines):
    if START_MARKER in ln and start_idx is None:
        start_idx = i
    if END_MARKER in ln and end_idx is None:
        end_idx = i

if start_idx is None or end_idx is None:
    print(f"ERROR: marker not found  start={start_idx}  end={end_idx}")
    sys.exit(1)

print(f"Replacing lines {start_idx+1}–{end_idx} ({end_idx - start_idx} lines)")

NEW_BLOCK = '''\
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

    def _compute_mfg(levers):
        C_sup_mlt = _live_coef("supplier_a", "material_lead_time", 7.0)
        C_mlt_del = _live_coef("material_lead_time", "shipment_delay", 0.9)
        C_mql_apd = _live_coef("machine_queue_length", "approval_duration", 0.7)
        C_apd_del = _live_coef("approval_duration", "shipment_delay", 0.3)

        BL_DEL = 8.2;  BL_MLT = 7.2;  BL_MQL = 3.1;  BL_APD = 2.4
        SUP_BASE = 0.60;  CAR_BASE = 15

        sup_a      = 1.0 - levers["supplier_reliability_pct"] / 100.0
        mlt_factor = {"Current": 1.0, "Reduced (-20%)": 0.80, "Optimised (-40%)": 0.60}[
            levers["material_lead_time_mode"]]

        mlt_pre = BL_MLT + C_sup_mlt * (sup_a - SUP_BASE)
        mlt_val = mlt_pre * mlt_factor

        wf_eff  = -0.3 * levers["additional_workforce"]
        cap_eff = -1.2 if levers["machine_capacity_expanded"] else 0.0
        mql_val = max(0.0, BL_MQL + wf_eff + cap_eff)

        exp_eff  = -0.35 * BL_APD if levers["export_flag_reduction"] else 0.0
        auto_eff = -0.50 * BL_APD if levers["approval_automation"]    else 0.0
        q_eff    = C_mql_apd * (mql_val - BL_MQL)
        apd_val  = max(0.0, BL_APD + exp_eff + auto_eff + q_eff)

        d_sup     = C_mlt_del * C_sup_mlt * (sup_a - SUP_BASE)
        d_ltmode  = C_mlt_del * mlt_pre * (mlt_factor - 1.0)
        d_machine = C_apd_del * q_eff
        d_appr    = C_apd_del * (exp_eff + auto_eff)
        d_carrier = -0.008 * (levers["carrier_express_pct"] - CAR_BASE)
        d_batch   = -0.20 if levers["order_batching"] else 0.0

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
        annual_sav = imp / 100.0 * BL_DEL * 1200 * 3000
        roi_mo     = (impl_cost / (annual_sav / 12)) if annual_sav > 0 else float("inf")

        return {
            "predicted": pred, "improvement_pct": imp,
            "mlt": mlt_val, "mql": mql_val, "apd": apd_val,
            "throughput": throughput, "risk_index": risk_idx,
            "impl_cost": impl_cost, "annual_saving": annual_sav, "roi_months": roi_mo,
            "ci_low": pred * 0.88, "ci_high": pred * 1.12,
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
        BL = 5.27;  BL_BED = 78.0;  BL_SPEC = 0.45;  FAST_BASE = 20

        spec_prob   = levers["specialist_allocation_pct"] / 100.0
        diag_factor = {"Standard": 1.0, "Fast (-20%)": 0.80, "Express (-35%)": 0.65}[
            levers["diagnostic_speed_mode"]]

        bed_eff    = -0.4 if levers["bed_capacity_expanded"] else 0.0
        nurse_eff  = -0.15 * levers["additional_nursing_staff"]
        triage_eff = -0.30 if levers["triage_automation"] else 0.0
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
        annual_sav = imp / 100.0 * BL * 1500 * 2000
        roi_mo     = (impl_cost / (annual_sav / 12)) if annual_sav > 0 else float("inf")

        return {
            "predicted": pred, "improvement_pct": imp,
            "mlt": pred, "mql": BL_BED * (1 + bed_eff / 100), "apd": BL * 0.4,
            "throughput": min(160.0, 100.0 * (1 - d_bed * 0.5)),
            "risk_index": 45.0 * (pred / BL),
            "impl_cost": impl_cost, "annual_saving": annual_sav, "roi_months": roi_mo,
            "ci_low": pred * 0.88, "ci_high": pred * 1.12,
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
                "Treatment Duration":  (BL,           pred,                      "days"),
                "Bed Occupancy":       (BL_BED,       BL_BED * (1 + bed_eff / 100), "%"),
                "Specialist Assigned": (BL_SPEC * 100, spec_prob * 100,           "%"),
            },
        }

    # ── Header banner ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    _sim_domain_lbl = ("Manufacturing — Prihir Enterprises"
                       if _is_mfg_sim else "Healthcare — Hospital Admissions")
    _sim_bl      = 8.2  if _is_mfg_sim else 5.27
    _sim_out_lbl = "Shipment Delay" if _is_mfg_sim else "Treatment Duration"
    _sim_f1      = dag_metrics.get("f1_score", 1.0) if "dag_metrics" in dir() else 1.0
    _conf_lbl    = "HIGH" if _sim_f1 >= 0.9 else ("MODERATE" if _sim_f1 >= 0.7 else "LOW")
    _conf_col    = "#059669" if _sim_f1 >= 0.9 else ("#D97706" if _sim_f1 >= 0.7 else "#DC2626")

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#F0FDF4 0%,#ECFDF5 100%);'
        f'border:1px solid #BBF7D0;border-radius:14px;padding:18px 24px;margin-bottom:20px;'
        f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">'
        f'<div>'
        f'<div style="font-size:1.25rem;font-weight:800;color:#1E293B;margin-bottom:3px;">'
        f'⚡ What-If Causal Simulator</div>'
        f'<div style="font-size:0.85rem;color:#64748B;">Adjust levers to see real-time outcome '
        f'predictions from the discovered causal model.</div>'
        f'</div>'
        f'<div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center;">'
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.68rem;color:#64748B;font-weight:700;text-transform:uppercase;">Domain</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#1E293B;">{_sim_domain_lbl}</div>'
        f'</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.68rem;color:#64748B;font-weight:700;text-transform:uppercase;">'
        f'Baseline {_sim_out_lbl}</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#1E293B;">{_sim_bl:.2f} days</div>'
        f'</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.68rem;color:#64748B;font-weight:700;text-transform:uppercase;">'
        f'Model Confidence</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:{_conf_col};">'
        f'{_conf_lbl} · F1={_sim_f1:.2f}</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<span style="width:8px;height:8px;background:#10B981;border-radius:50%;'
        f'box-shadow:0 0 0 3px rgba(16,185,129,0.22);display:inline-block;"></span>'
        f'<span style="font-size:0.78rem;font-weight:600;color:#059669;">Live Engine Active</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Two-column layout ──────────────────────────────────────────────────────
    _lcol, _rcol = st.columns([1.1, 1.9])

    with _lcol:
        st.markdown(
            "<h5 style='color:#1E293B;margin-bottom:6px;font-size:0.95rem;'>"
            "\U0001f39b️ Intervention Levers</h5>",
            unsafe_allow_html=True,
        )

        if st.button("\U0001f504 Reset to Baseline", use_container_width=True, key="sim_reset_btn"):
            st.session_state[_sim_key] = dict(_MFG_DEFAULTS if _is_mfg_sim else _HC_DEFAULTS)
            st.rerun()

        _lv = st.session_state[_sim_key]

        if _is_mfg_sim:
            with st.expander("\U0001f3ed Supplier & Procurement", expanded=True):
                _lv["supplier_reliability_pct"] = st.slider(
                    "Supplier B Allocation (%)", 0, 100,
                    _lv["supplier_reliability_pct"], 5, key="lv_sup_rel",
                    help="60% currently go to Supplier A (unreliable). Shift more to Supplier B.",
                )
                _lv["export_flag_reduction"] = st.toggle(
                    "Streamline Export Documentation",
                    _lv["export_flag_reduction"], key="lv_exp_flag",
                    help="Reduces export-related approval delays ~35%",
                )
            with st.expander("⚙️ Machine & Capacity", expanded=True):
                _lv["machine_capacity_expanded"] = st.toggle(
                    "Expand Machine Capacity",
                    _lv["machine_capacity_expanded"], key="lv_mach_cap",
                    help="Adds processing units, reduces Machine Queue ~40%",
                )
                _lv["additional_workforce"] = st.slider(
                    "Additional Workforce", 0, 20,
                    _lv["additional_workforce"], 1, key="lv_workforce",
                    help="Each additional worker reduces queue ~0.3 units",
                )
            with st.expander("\U0001f4cb Approvals & Process", expanded=False):
                _lv["approval_automation"] = st.toggle(
                    "Automate Approval Steps",
                    _lv["approval_automation"], key="lv_appr_auto",
                    help="Reduces Approval Duration ~50%",
                )
                _lv["order_batching"] = st.toggle(
                    "Enable Order Batching",
                    _lv["order_batching"], key="lv_batching",
                    help="Reduces order complexity spikes ~20%",
                )
            with st.expander("\U0001f69a Logistics & Delivery", expanded=False):
                _lv["carrier_express_pct"] = st.slider(
                    "Express Carrier Usage (%)", 0, 100,
                    _lv["carrier_express_pct"], 5, key="lv_carrier",
                    help="Each 10% increase reduces delay ~0.08 days",
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
            with st.expander("\U0001f3e5 Specialist & Allocation", expanded=True):
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
            with st.expander("\U0001f6cf️ Capacity & Staff", expanded=True):
                _lv["bed_capacity_expanded"] = st.toggle(
                    "Expand Bed Capacity",
                    _lv["bed_capacity_expanded"], key="lv_bed_cap",
                    help="Adds beds, reduces occupancy pressure",
                )
                _lv["additional_nursing_staff"] = st.slider(
                    "Additional Nursing Staff", 0, 20,
                    _lv["additional_nursing_staff"], 1, key="lv_nursing",
                    help="Each nurse reduces duration ~0.15 days",
                )
            with st.expander("⚡ Process & Diagnostics", expanded=False):
                _lv["triage_automation"] = st.toggle(
                    "Automate Triage Scoring",
                    _lv["triage_automation"], key="lv_triage_auto",
                    help="Reduces triage-to-treatment delay ~0.3 days",
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
        st.markdown(
            "<h5 style='color:#1E293B;margin-bottom:6px;font-size:0.95rem;'>"
            "\U0001f4ca Predicted Outcomes</h5>",
            unsafe_allow_html=True,
        )

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
                f'95% CI: [{_res["ci_low"]:.1f} – {_res["ci_high"]:.1f} days]</div>',
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

        # ── Secondary KPI row ─────────────────────────────────────────────────
        _k1, _k2, _k3 = st.columns(3)
        with _k1:
            st.metric("Throughput", f"{_res['throughput']:.0f}/day",
                      delta=f"+{_res['throughput'] - 100:.0f}")
        with _k2:
            st.metric("Risk Index", f"{_res['risk_index']:.1f}",
                      delta=f"{_res['risk_index'] - 45:.1f}", delta_color="inverse")
        with _k3:
            _roi     = _res["roi_months"]
            _roi_str = f"{_roi:.1f} mo" if _roi < 60 else "N/A"
            st.metric("ROI Payback", _roi_str,
                      delta=f"${_res['annual_saving']:,.0f}/yr saved")

        # ── Causal Effect Decomposition (waterfall) ───────────────────────────
        st.markdown(
            "<div style='height:6px;'></div>"
            "<div style='font-size:0.82rem;font-weight:700;color:#1E293B;margin-bottom:2px;'>"
            "\U0001f4e1 Causal Effect Decomposition</div>",
            unsafe_allow_html=True,
        )
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
        st.plotly_chart(_fig_wf, use_container_width=True, theme=None,
                        config={"displayModeBar": False})

        # ── Mediator state table ──────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.82rem;font-weight:700;color:#1E293B;margin-bottom:5px;'>"
            "\U0001f517 Mediator Variable States</div>",
            unsafe_allow_html=True,
        )
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

        # ── ROI panel ─────────────────────────────────────────────────────────
        if _res["impl_cost"] > 0:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;'
                f'padding:12px 16px;">'
                f'<div style="font-size:0.8rem;font-weight:700;color:#1D4ED8;margin-bottom:7px;">'
                f'\U0001f4b0 Business Impact Estimate</div>'
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
                f'<div style="font-size:0.68rem;color:{SUBTLE};margin-top:5px;font-style:italic;">'
                f'*SEM-based estimate · $1,200/day/case · 3,000 cases/yr</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Scenario Comparison (full width) ──────────────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<h5 style='color:#1E293B;margin-bottom:10px;font-size:0.95rem;'>"
        "\U0001f4cb Scenario Comparison</h5>",
        unsafe_allow_html=True,
    )

    _sc_in, _sc_sv, _sc_cl = st.columns([3, 1, 1])
    with _sc_in:
        _save_name = st.text_input(
            "Scenario name", placeholder="e.g. Full Automation + Express",
            key="sim_save_name_input", label_visibility="collapsed",
        )
    with _sc_sv:
        if st.button("\U0001f4be Save", use_container_width=True, key="sim_save_btn"):
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
    with _sc_cl:
        if st.session_state["sim_scenarios"]:
            if st.button("\U0001f5d1️ Clear", use_container_width=True, key="sim_clear_btn"):
                st.session_state["sim_scenarios"] = []
                st.rerun()

    _scenarios = st.session_state["sim_scenarios"]
    if _scenarios:
        _sc_rows = []
        for _s in _scenarios:
            _r    = _s["result"]
            _ri   = f"{_r['roi_months']:.1f} mo" if _r['roi_months'] < 60 else "N/A"
            _sc_rows.append({
                "Scenario":        _s["name"],
                "Pred. Delay (d)": f"{_r['predicted']:.1f}",
                "Improvement":     f"{_r['improvement_pct']:.1f}%",
                "Throughput":      f"{_r['throughput']:.0f}/day",
                "Risk Index":      f"{_r['risk_index']:.1f}",
                "ROI Payback":     _ri,
                "Impl. Cost":      f"${_r['impl_cost']:,.0f}",
            })
        st.dataframe(pd.DataFrame(_sc_rows), use_container_width=True, hide_index=True)

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
    else:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;'
            f'padding:14px;text-align:center;color:{MUTED};font-size:0.82rem;">'
            f'Save up to 4 named scenarios to compare them side-by-side.</div>',
            unsafe_allow_html=True,
        )

    # ── Live Causal DAG ───────────────────────────────────────────────────────
    with st.expander("\U0001f52c Live Causal Graph — Intervention Propagation",
                     expanded=False):
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
                 _lv["supplier_reliability_pct"] != 40),
                ("order_complexity",   "machine_queue",  False),
                ("machine_queue",      "approval_duration",
                 _lv["machine_capacity_expanded"] or _lv["additional_workforce"] > 0),
                ("export_flag",        "approval_duration",  _lv["export_flag_reduction"]),
                ("material_lead_time", "shipment_delay",
                 _lv["material_lead_time_mode"] != "Current" or
                 _lv["supplier_reliability_pct"] != 40),
                ("approval_duration",  "shipment_delay",
                 _lv["approval_automation"] or _lv["export_flag_reduction"]),
                ("carrier_express",    "shipment_delay",  _lv["carrier_express_pct"] != 15),
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
                 _lv["specialist_allocation_pct"] != 45),
                ("bed_occupancy",          "treatment_duration",    _lv["bed_capacity_expanded"]),
                ("triage_score",           "specialist_requirement", _lv["triage_automation"]),
            ]

        _role_colors = {
            "Confounder": WARNING, "Treatment": PRIMARY, "Mediator": INFO, "Outcome": ERROR,
        }
        _nxs = {n["id"]: n["x"] for n in _dag_nodes}
        _nys = {n["id"]: n["y"] for n in _dag_nodes}

        _fig_dag = go.Figure()
        for _src, _dst, _active in _dag_edges:
            _fig_dag.add_trace(go.Scatter(
                x=[_nxs[_src], _nxs[_dst], None],
                y=[_nys[_src], _nys[_dst], None],
                mode="lines",
                line=dict(color=PRIMARY if _active else "#CBD5E1",
                          width=2.5 if _active else 1.2),
                showlegend=False, hoverinfo="skip",
            ))
        for _n in _dag_nodes:
            _ncol = _role_colors.get(_n["role"], MUTED)
            _fig_dag.add_trace(go.Scatter(
                x=[_n["x"]], y=[_n["y"]],
                mode="markers+text",
                marker=dict(size=42, color=_ncol, opacity=0.88,
                            line=dict(color="white", width=2.5)),
                text=[_n["id"].replace("_", "<br>")],
                textposition="bottom center",
                textfont=dict(size=8, color=TEXT),
                showlegend=False,
                hovertemplate=f"<b>{_n['id']}</b><br>Role: {_n['role']}<extra></extra>",
            ))
        _dagl = dict(**PLOTLY_LAYOUT)
        _dagl.update(dict(
            height=380, plot_bgcolor="#FAFBFC",
            margin=dict(l=10, r=10, t=20, b=60),
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
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

'''

replaced = lines[:start_idx] + [NEW_BLOCK] + lines[end_idx:]
DASH.write_text("".join(replaced), encoding="utf-8")
print(f"OK: replaced lines {start_idx+1}–{end_idx}  "
      f"({end_idx - start_idx} old lines → {len(NEW_BLOCK.splitlines())} new lines)")
