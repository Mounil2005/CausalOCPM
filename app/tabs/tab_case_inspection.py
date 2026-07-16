# ── CASE INSPECTION TAB ──────────────────────────────────────────────────
# Split out of Model & Impact: this is the only content in the app whose
# numbers change depending on which single case you pick (SHAP attribution
# for one order/patient). Model Performance, Sensitivity to Unmeasured
# Confounding, Recovery Visualization, and CATE all stayed in Model & Impact
# because they're population/model-level diagnostics that don't move when
# you change the case selector — putting them here would have been a
# naming mismatch (opening "Case Inspection" and seeing numbers that don't
# react to the case you're inspecting).
from src.phase5_attribution import (explain_case, get_attribution_summary,
                                     explain_limitation)

outcome_var   = cfg["outcome_var"]
outcome_label = cfg["outcome_label"]

if is_custom:
    accuracy_disclaimer(custom_confidence, len(df), custom_quality.get("score", 0))

id_cols     = ["order_id", "patient_id"]
case_id_col = next((c for c in id_cols if c in df.columns), None)
case_ids    = df[case_id_col].tolist() if case_id_col else [f"Case_{i}" for i in range(len(df))]
_case_pool  = case_ids[:200]

# ── Tab intro ────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:linear-gradient(135deg,#FDF2F8,#FFFFFF);border:1px solid #FBCFE8;'
    f'border-left:4px solid #EC4899;border-radius:14px;padding:20px 26px;margin-bottom:24px;'
    f'box-shadow:0 4px 12px rgba(0,0,0,0.03);">'
    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
    f'<span style="width:30px;height:30px;border-radius:9px;background:#EC489922;display:flex;'
    f'align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;">🔬</span>'
    f'<span style="font-size:0.85rem;font-weight:800;color:#EC4899;text-transform:uppercase;'
    f'letter-spacing:0.08em;">Case Inspector</span>'
    f'</div>'
    f'<div style="font-size:0.95rem;color:#334155;line-height:1.5;">'
    f'Drill into any individual {("order" if domain == "manufacturing" else "patient")} to see exactly '
    f'why the model predicted what it did — SHAP attribution breaks the outcome down into the specific '
    f'factors that pushed it above or below the population average, split into what\'s controllable '
    f'versus structural.</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Case navigation — Previous / Next / Jump to Highest-Risk, plus direct
# pick. Session state is set before the selectbox is instantiated (and
# without also passing `index=`) so Streamlit doesn't warn about a widget
# key being set two different ways.
if "case_selector" not in st.session_state or st.session_state["case_selector"] not in _case_pool:
    st.session_state["case_selector"] = _case_pool[0]

# Stepper layout: ◀ / selector / ▶ read as one paginator unit (arrows
# flanking the dropdown they control) instead of four disconnected boxes
# spread across the row. "Jump to Highest-Risk Case" is kept visually
# separate and styled as the app's primary action color, since it's a
# distinct shortcut, not part of stepping through cases one at a time.
_cur_pos = _case_pool.index(st.session_state["case_selector"])
st.markdown(
    f'<div style="font-size:0.72rem;font-weight:700;color:{MUTED};text-transform:uppercase;'
    f'letter-spacing:0.05em;margin-bottom:4px;">Case {_cur_pos + 1} of {len(_case_pool)}'
    + (f' &nbsp;·&nbsp; showing first {len(_case_pool):,} of {len(case_ids):,} cases' if len(case_ids) > 200 else '')
    + '</div>',
    unsafe_allow_html=True,
)
_nav_prev, _nav_pick, _nav_next, _nav_risk = st.columns([0.6, 3.4, 0.6, 2.4])
# Columns fix each widget's on-screen position regardless of the order
# code runs in, but session_state["case_selector"] can't be written after
# the selectbox with that key has been instantiated THIS run — so every
# button's click is created and checked before the selectbox line below,
# even though ◀ and ▶ visually sit on either side of it.
with _nav_prev:
    _clicked_prev = st.button("◀", use_container_width=True, help="Previous case")
with _nav_next:
    _clicked_next = st.button("▶", use_container_width=True, help="Next case")
with _nav_risk:
    _clicked_risk = st.button("🎯 Jump to Highest-Risk Case", use_container_width=True, type="primary")

if _clicked_prev:
    st.session_state["case_selector"] = _case_pool[(_cur_pos - 1) % len(_case_pool)]
    st.rerun()
if _clicked_next:
    st.session_state["case_selector"] = _case_pool[(_cur_pos + 1) % len(_case_pool)]
    st.rerun()
if _clicked_risk:
    _pool_vals = df[outcome_var].iloc[:len(_case_pool)].to_numpy()
    st.session_state["case_selector"] = _case_pool[int(_pool_vals.argmax())]
    st.rerun()

with _nav_pick:
    selected_case = st.selectbox(
        "Select Case", options=_case_pool, key="case_selector", label_visibility="collapsed",
    )
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

def _ci_section_header(icon_bg, icon, title, subtitle=None):
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:9px; margin-bottom:4px;">'
        f'<div style="width:26px; height:26px; border-radius:7px; background:{icon_bg}; '
        f'display:flex; align-items:center; justify-content:center; font-size:0.8rem; flex-shrink:0;">{icon}</div>'
        f'<span style="color:#1E293B; font-size:0.95rem; font-weight:700;">{title}</span>'
        f'</div>'
        + (f'<div style="color:#64748B; font-size:0.78rem; margin:0 0 12px 35px;">{subtitle}</div>' if subtitle else '<div style="height:8px;"></div>'),
        unsafe_allow_html=True,
    )

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

    # Where this case sits in the overall population — higher outcome value
    # means worse (longer delay/stay), so "worse than X%" tracks directly
    # with the percentile of the raw outcome column.
    _pct_worse = float((df[outcome_var] < actual).mean() * 100)
    if _pct_worse >= 75:
        risk_tier, risk_color = "High Risk", ERROR
    elif _pct_worse >= 40:
        risk_tier, risk_color = "Moderate Risk", WARNING
    else:
        risk_tier, risk_color = "Low Risk", SUCCESS

    # Top contributor
    top_row = expl.iloc[expl["shap_value"].abs().idxmax()]
    top_contributor = top_row["feature"].replace("_", " ").title()

    # Executive Interpretation Banner
    exec_text = (
        f"Case <b>{selected_case}</b> achieved an outcome of {actual:.2f} {outcome_label}, {performance}. "
        f"<b>{top_contributor}</b> was the dominant contributor to this outcome. "
        f"Additional interventions targeting controllable factors could further improve the outcome by approximately {attrib_summary['max_reducible_delay']:.2f} {outcome_label}."
    )
    # Banner tint follows whether this case is actually better or worse than
    # baseline — it previously always rendered green, which read as "good
    # news" even on cases that underperformed the population average.
    _exec_bg = "#F0FDF4" if diff < 0 else "#FFFBEB"
    st.markdown(
        f'<div style="background:{_exec_bg}; border-left:4px solid {status_color}; padding:20px; border-radius:4px; margin-bottom:32px; box-shadow:0 2px 8px rgba(0,0,0,0.02);">'
        f'<div style="color:{status_color}; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Executive Interpretation</div>'
        f'<div style="color:#1E293B; font-size:1.05rem; line-height:1.6;">{exec_text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Case Snapshot — 4 tiles: outcome, complexity, treatment, and now
    # population percentile / risk tier (new — lets you tell at a glance
    # whether this case is actually unusual or a routine one).
    complexity_col = "order_complexity" if domain == "manufacturing" else "patient_complexity"
    comp_val = df[complexity_col].iloc[case_idx] if complexity_col in df.columns else 0
    comp_text = "High" if comp_val > 5 else "Low"

    treat_label = cfg["treatment_options"].get(cfg["treatment_var"], cfg["treatment_var"]) if "treatment_options" in cfg else "Treatment"
    treat_val = "Yes" if int(df[cfg["treatment_var"]].iloc[case_idx]) == 1 else "No" if cfg["treatment_var"] in df.columns else "N/A"

    snap_html = (
        f'<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px;">'
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
        f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid {risk_color}; padding:16px; border-radius:10px;">'
        f'<div style="color:#64748B; font-size:0.8rem; font-weight:700; text-transform:uppercase;">Population Percentile</div>'
        f'<div style="font-size:1.8rem; font-weight:800; color:{TEXT}; margin-top:4px;">{_pct_worse:.0f}%</div>'
        f'<div style="font-size:0.85rem; font-weight:600; color:{risk_color}; margin-top:4px;">{risk_tier} · worse than {_pct_worse:.0f}%</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(snap_html, unsafe_allow_html=True)

    # Outcome Attribution (Hero Visualization) — wrapped in a bordered card
    # (st.container(border=True), the same primitive tab_model_impact.py
    # uses for its own waterfall) so this reads as one section instead of a
    # heading and a chart floating loose on the page background.
    with st.container(border=True):
        _ci_section_header(
            "#FCE7F3", "📉", "Why Did This Outcome Occur?",
            "SHAP decomposition — how each factor pushed this case's outcome away from the population average",
        )

        _wf_labels = ["Population Average"] + [f.replace("_", " ").title() for f in features] + ["Case Prediction"]
        _wf_values = [baseline] + shap_vals + [0]
        # Explicit formatted text per bar instead of texttemplate — texttemplate
        # on go.Waterfall doesn't reliably apply number formatting across
        # Plotly versions and was rendering raw, unrounded floats.
        _wf_text = [f"{baseline:.2f}"] + [f"{v:+.2f}" for v in shap_vals] + [f"{predicted:.2f}"]

        fig_wf = go.Figure(go.Waterfall(
            x=_wf_labels,
            y=_wf_values,
            measure=["absolute"] + ["relative"] * len(shap_vals) + ["total"],
            connector=dict(line=dict(color=BORDER, width=1, dash="dot")),
            increasing=dict(marker_color=ERROR),
            decreasing=dict(marker_color=SUCCESS),
            totals=dict(marker_color="#334155"),
            text=_wf_text, textposition="outside",
            textfont=dict(size=12),
        ))
        # No inline annotation on the hline itself — at low actual/baseline
        # ratios it landed on top of the first bar's own value label. A
        # caption below the chart says the same thing without any overlap risk.
        fig_wf.add_hline(y=actual, line_dash="dash", line_color="#94A3B8")
        # Explicit y-range with headroom — without it the tallest bar (usually
        # Population Average) touched the very top of the plot with no room
        # for its own text label, and the dashed "Actual" line could sit flush
        # against the top edge too.
        _y_top = max(_wf_values + [actual, predicted, baseline]) * 1.2
        _wfl = dict(**PLOTLY_LAYOUT)
        _wfl.update(dict(
            yaxis={**PLOTLY_LAYOUT.get("yaxis", {}), "title": outcome_label, "title_font": dict(size=13),
                   "range": [0, _y_top]},
            xaxis={**PLOTLY_LAYOUT.get("xaxis", {}), "tickangle": -30, "tickfont": dict(size=11), "automargin": True},
            height=440,
            margin=dict(l=70, r=30, t=40, b=100),
            showlegend=False,
        ))
        fig_wf.update_layout(**_wfl)
        try:
            st.plotly_chart(fig_wf, use_container_width=True, theme=None, config={'displayModeBar': False})
        except Exception as _e:
            st.error(f"Chart error: {_e}")
        st.markdown(
            f'<div style="font-size:0.78rem; color:#94A3B8; margin-top:-8px;">'
            f'<span style="display:inline-block; width:16px; border-top:2px dashed #94A3B8; '
            f'vertical-align:middle; margin-right:6px;"></span>'
            f'Dashed line marks the actual recorded outcome ({actual:.2f} {outcome_label}) — the bars build up '
            f'the model\'s <i>predicted</i> outcome ({predicted:.2f}) instead, so a gap between the two is the '
            f'model\'s residual error for this case.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # Actionability Insights
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

    # Key Insight Summary
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

    # Technical Evidence
    with st.expander("🔍 Detailed Attribution Analysis"):
        st.markdown("Raw SHAP values and attribution categories supporting the executive summary.")
        detail = expl[["feature", "attribution", "shap_value", "feature_value"]].copy()
        detail.columns = ["Feature", "Attribution", "SHAP Value", "Feature Value"]

        tbl_rows = ""
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

# Sensitivity to Unmeasured Confounding — every case's SHAP attribution
# above rests on the same Double ML causal effect, so "how much should I
# trust the causal story behind this case" belongs here too. Reuses the
# exact function from Model & Impact (via shared exec globals) rather
# than duplicating ~70 lines of chart/metric code.
_render_sensitivity_section()

# Methodological Foundation
with st.expander("📚 Methodological Foundation"):
    st.markdown(explain_limitation(include_citation=True))
