def render_copilot():
    import streamlit as st
    import time
    import numpy as np

    # ── 10x SAAS CSS INJECTION ───────────────────────────────────────────────
    # Kept as ONE single <style> tag — splitting this into several concatenated
    # <style> blocks in one st.markdown() call was tried in the tab-bar redesign
    # and silently truncated every block but the first (a real bug, root-caused
    # by inspecting computed styles, not guessed at) — so everything here stays
    # in one block.
    st.markdown("""
        <style>
        /* 1. Global Page Background & Font Smoothing */
        .stApp {
            background-color: #F8FAFC !important;
            -webkit-font-smoothing: antialiased;
        }

        /* 2. Chat Input Glow (Phase 3/16) */
        [data-testid="stChatInput"] {
            position: relative !important;
            border-radius: 20px !important;
            border: 1px solid #DDD6FE !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04) !important;
            background: #FFFFFF !important;
            padding: 4px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        /* Decorative leading sparkle — purely visual, matching the reference;
           there's no real file-attach/voice-input capability behind
           Streamlit's chat_input to back real icons for those, so only this
           one (which claims no functionality) was added. The icon's room is
           made via padding on the *textarea* below, not on this outer
           container — adding 40px of padding here previously shrank this
           box's available content width without Streamlit's own absolutely-
           positioned submit-button wrapper knowing about it (confirmed via
           getBoundingClientRect: its containing block stayed the old,
           wider size), so the button visibly overflowed past the pill's
           right edge. Leaving this box's own padding minimal avoids that. */
        [data-testid="stChatInput"]::before {
            content: "✦"; position: absolute; left: 16px; top: 50%;
            transform: translateY(-50%); color: #A78BFA; font-size: 1rem; pointer-events: none;
        }
        [data-testid="stChatInput"] textarea {
            padding-left: 32px !important;
        }
        [data-testid="stChatInputSubmitButton"] {
            background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
            border-radius: 14px !important;
            color: #FFFFFF !important;
            padding: 0 9px !important;
            height: 36px !important;
            gap: 4px !important;
            /* The button's own absolutely-positioned wrapper (added by
               Streamlit, no stable selector to target directly) still sits
               ~5px right of this pill's actual border — nudging the button
               itself left compensates without touching that wrapper. */
            position: relative !important;
            right: 5px !important;
        }
        [data-testid="stChatInputSubmitButton"] svg { width: 15px !important; height: 15px !important; }
        [data-testid="stChatInputSubmitButton"]:disabled { opacity: 0.4 !important; }
        [data-testid="stChatInputSubmitButton"]::after {
            content: "Send"; font-size: 0.8rem; font-weight: 700; color: #FFFFFF; white-space: nowrap;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #7C3AED !important;
            box-shadow: 0 0 0 4px rgba(124,58,237,0.15), 0 8px 24px rgba(0,0,0,0.06) !important;
            transform: translateY(-2px);
        }
        /* BaseWeb's own textarea wrapper draws its own border independently
           of the outer stChatInput div above — confirmed via computed style
           inspection, and matching its color to the outer pill (two earlier
           fix attempts) still didn't fix it: the inner wrapper's own
           getBoundingClientRect() showed it is a DIFFERENT SHAPE at a
           different size than the outer pill — 8px square corners nested
           inside the outer's 20px pill curve, and its right edge actually
           overflows 5px past the outer container's right edge. Two nested
           boxes of different shape/size will look like two boxes no matter
           what color either border is. The only fix that holds regardless of
           BaseWeb's internal geometry is to never let the inner wrapper draw
           a visible border at all, in any state, and let the outer pill be
           the only border/ring the user ever sees. */
        [data-testid="stChatInput"] [data-baseweb="textarea"] {
            border-color: transparent !important;
        }

        /* 3. Base Chat Message Styling */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            padding: 12px 0 !important;
            animation: fade-in-up 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        @keyframes fade-in-up {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16,185,129, 0.4); }
            70% { box-shadow: 0 0 0 8px rgba(16,185,129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16,185,129, 0); }
        }

        /* 4. User Bubble (Right Aligned, Dark) */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            flex-direction: row-reverse;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
            margin-left: 1rem;
            margin-right: 0;
            background: #E2E8F0 !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
            background: linear-gradient(135deg, #0F172A, #1E293B);
            color: #FFFFFF !important;
            padding: 14px 20px;
            border-radius: 24px 24px 4px 24px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
            color: #FFFFFF !important;
            margin: 0;
            font-size: 0.95rem;
            font-weight: 500;
        }

        /* 5. Assistant Bubble (Left Aligned, White Card) */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
            background: #FFFFFF;
            color: #1E293B !important;
            padding: 24px 32px;
            border-radius: 4px 24px 24px 24px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.04);
            border: 1px solid #E2E8F0;
            width: 100%;
        }

        /* 6. AI Avatar Styling */
        [data-testid="chatAvatarIcon-assistant"] {
            background: linear-gradient(135deg, #10B981, #059669) !important;
            box-shadow: 0 4px 12px rgba(16,185,129,0.3) !important;
            color: white !important;
        }

        /* 7. Notion-like Markdown Formatting inside Assistant Bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h3 {
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748B !important;
            margin-top: 1.8rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 4px;
            font-weight: 800;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h3:first-of-type {
            margin-top: 0;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
            font-size: 0.95rem;
            line-height: 1.6;
            color: #334155;
            margin-bottom: 1rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li {
            font-size: 0.95rem;
            line-height: 1.6;
            color: #334155;
            margin-bottom: 0.5rem;
        }

        /* 8. Suggestion Chips (main empty-state cards) */
        .saas-suggestion-btn button {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            color: #1E293B !important;
            font-weight: 600 !important;
            padding: 16px 20px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            width: 100%;
            text-align: left !important;
            justify-content: flex-start !important;
        }
        .saas-suggestion-btn button:hover {
            border-color: #10B981 !important;
            color: #059669 !important;
            box-shadow: 0 8px 16px rgba(16,185,129,0.08) !important;
            transform: translateY(-2px) !important;
        }
        .saas-suggestion-btn--sm button {
            padding: 8px 14px !important;
            font-size: 0.85rem !important;
            border-radius: 10px !important;
        }
        .saas-suggestion-btn--warn button { border-left: 3px solid #D97706 !important; }
        .saas-suggestion-btn--info button { border-left: 3px solid #0284C7 !important; }
        .saas-suggestion-btn--success button { border-left: 3px solid #059669 !important; }

        /* 9. Popover trigger buttons (currently just "More Tools"). A
           `.cop-icon-btn` wrapper div (opened via st.markdown, closed after
           the st.popover call) was tried here first — inspecting the live
           DOM showed it renders as its own empty, self-closed div, exactly
           like the capability-card bug: st.popover isn't a plain st.button,
           so there's no st.container(border=True)-style fix available for
           it either. Rather than fight that, every popover trigger button
           on this tab gets a shared style scoped to real `st.popover`
           ancestry (`[data-testid="stPopover"]`) rather than a broken
           wrapper div — st.popover isn't used anywhere else in the app, so
           this can't leak onto other tabs' plain st.button widgets the way
           an unscoped `button[data-testid=...]` selector would. */
        [data-testid="stPopover"] button[data-testid="baseButton-secondary"] { border-radius: 12px !important; }
        [data-testid="stPopover"] button[data-testid="baseButton-secondary"] svg { display: none !important; }

        /* 11. Capability cards — the bordered box itself is a real
           st.container(border=True), not a raw HTML div (see the note by
           where it's built), so only the content pieces below need classes.
           A :has()-based left-border-via-invisible-marker trick was tried
           here first, but Streamlit nests st.container(border=True) inside
           at least two OTHER elements sharing the same
           stVerticalBlockBorderWrapper test-id (one from the nested
           st.columns([6,1]) inside it, one from the tab panel itself) —
           :has() matches every ancestor that contains the marker, not just
           the nearest, so it colored the entire tab panel's outer edge as
           one continuous bar instead of 4 card borders. Fixed by rendering
           a real, visible accent bar as actual content inside the
           container instead of trying to recolor an ancestor. */
        .cop-card-accent-bar { height: 4px; border-radius: 3px; margin: -4px 0 14px 0; }
        .cop-card-row { display: flex; align-items: flex-start; gap: 16px; }
        .cop-card-icon {
            width: 56px; height: 56px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
        }
        .cop-card-title { font-size: 1rem; font-weight: 800; color: #0F172A; margin-bottom: 3px; }
        .cop-card-desc { font-size: 0.85rem; color: #64748B; margin-bottom: 10px; }
        .cop-card-tags { display: flex; gap: 8px; flex-wrap: wrap; }
        .cop-tag {
            font-size: 0.68rem; font-weight: 700; padding: 3px 10px; border-radius: 8px;
        }
        .cop-card-arrow button {
            width: 40px !important; height: 40px !important; border-radius: 50% !important;
            padding: 0 !important; font-weight: 800 !important; font-size: 1.1rem !important;
            border: none !important; margin-top: 6px;
        }
        .cop-card-arrow--warn button    { background: #FEF3C7 !important; color: #D97706 !important; }
        .cop-card-arrow--purple button  { background: #EDE9FE !important; color: #7C3AED !important; }
        .cop-card-arrow--blue button    { background: #DBEAFE !important; color: #2563EB !important; }
        .cop-card-arrow--success button { background: #D1FAE5 !important; color: #059669 !important; }
        .cop-card-arrow button:hover { filter: brightness(0.95); }

        /* 12. Secondary tool-chip row — a `.cop-tool-chip` wrapper div was
           tried here too (border/nowrap on `button` descendant) but, like
           `.cop-icon-btn`, inspection showed it renders as its own empty
           div — Streamlit's default secondary-button look (white, bordered,
           rounded) already reads fine here, so short one-line labels
           (picked in the tool_chips list below) were the real fix instead
           of a non-functional CSS class. */

        /* 13. Bottom quick-prompt chips (above the sticky chat input) */
        .cop-bottom-chip button {
            background: #F8FAFC !important; border: 1px solid #E2E8F0 !important;
            border-radius: 20px !important; color: #475569 !important; font-weight: 600 !important;
            font-size: 0.78rem !important; padding: 7px 16px !important;
        }
        .cop-bottom-chip button:hover { border-color: #A78BFA !important; color: #7C3AED !important; }
        </style>
    """, unsafe_allow_html=True)

    # ── INITIALIZE STATE ────────────────────────────────────────────────────────
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = []

    try:
        _active_key = st.secrets.get("CEREBRAS_API_KEY", "")
    except Exception:
        _active_key = ""

    _key_active = bool(_active_key)
    _cop_domain = domain.replace("_", " ").title()

    # Status badge reflects whether the last real API call actually succeeded,
    # not merely whether a key string is present — a bad/expired key or model
    # error still produces a normal-looking canned answer via the fallback
    # path, so key-presence alone would silently misreport "connected".
    _last_call_ok = st.session_state.get("copilot_last_call_ok")  # None = untested this session
    if not _key_active:
        _status_label, _status_bg, _status_fg, _status_border, _status_dot = (
            "OFFLINE FALLBACK", "#F1F5F9", "#475569", "#E2E8F0", "#94A3B8")
    elif _last_call_ok is True:
        _status_label, _status_bg, _status_fg, _status_border, _status_dot = (
            "CONNECTED: CEREBRAS LLM", "#ECFDF5", "#059669", "#A7F3D0", "#10B981")
    elif _last_call_ok is False:
        _status_label, _status_bg, _status_fg, _status_border, _status_dot = (
            "FALLBACK MODE — API ERROR", "#FFFBEB", "#B45309", "#FDE68A", "#F59E0B")
    else:
        _status_label, _status_bg, _status_fg, _status_border, _status_dot = (
            "CEREBRAS LLM READY", "#F8FAFC", "#475569", "#E2E8F0", "#94A3B8")

    # "Live pipeline data" reflects whether a real, non-empty pipeline is
    # actually available to ground answers in — not a decorative constant.
    _pipeline_live = bool(
        dag is not None and dag.number_of_nodes() > 0
        and df is not None and len(df) > 0
    )
    if _pipeline_live:
        _pipe_label, _pipe_bg, _pipe_fg, _pipe_border, _pipe_dot = (
            "LIVE PIPELINE DATA", "#F8FAFC", "#475569", "#E2E8F0", "#3B82F6")
    else:
        _pipe_label, _pipe_bg, _pipe_fg, _pipe_border, _pipe_dot = (
            "PIPELINE DATA UNAVAILABLE", "#FFFBEB", "#B45309", "#FDE68A", "#F59E0B")

    _copilot_active = _key_active and _pipeline_live

    # ── PHASE 2: HERO HEADER ──────────────────────────────────────────────────
    st.markdown(
        f"""
<div style="background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(12px); border: 1px solid #DDD6FE; border-radius: 20px; padding: 28px 32px; margin-bottom: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.02); position: relative; overflow: hidden;">
<div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #10B981, #3B82F6);"></div>
<svg style="position:absolute; top:0; right:0; width:60%; height:100%; opacity:0.5; pointer-events:none;" viewBox="0 0 600 180" preserveAspectRatio="none">
<path d="M0,120 C100,80 200,150 300,100 C400,60 500,130 600,90" fill="none" stroke="#A7F3D0" stroke-width="2"/>
<path d="M0,140 C120,110 220,170 320,120 C420,80 520,150 600,110" fill="none" stroke="#BFDBFE" stroke-width="2"/>
</svg>
<span style="position:absolute; top:28px; right:120px; color:#A78BFA; font-size:1.1rem;">✦</span>
<span style="position:absolute; top:64px; right:260px; color:#93C5FD; font-size:0.8rem;">✦</span>
<span style="position:absolute; top:100px; right:80px; color:#6EE7B7; font-size:0.9rem;">✦</span>
<div style="display: flex; justify-content: space-between; align-items: flex-start; position: relative;">
<div style="display: flex; align-items: center; gap: 16px;">
<div style="position: relative; flex-shrink: 0;">
<div style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, #10B981, #059669); box-shadow: 0 4px 12px rgba(16,185,129,0.3); display: flex; align-items: center; justify-content: center; font-size: 1.7rem;">🤖</div>
<div style="position:absolute; bottom:-4px; right:-4px; width:20px; height:20px; border-radius:50%; background:#7C3AED; border:2px solid #fff; display:flex; align-items:center; justify-content:center; font-size:0.6rem;">✨</div>
</div>
<div>
<h2 style="color: #0F172A; margin: 0 0 4px 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em;">Enterprise AI Copilot</h2>
<p style="color: #64748B; margin: 0 0 8px 0; font-size: 1.05rem; font-weight: 500;">Process Intelligence Platform</p>
<div style="display: inline-flex; align-items: center; gap: 7px; background: {'#ECFDF5' if _copilot_active else '#FFFBEB'}; color: {'#059669' if _copilot_active else '#B45309'}; border: 1px solid {'#A7F3D0' if _copilot_active else '#FDE68A'}; padding: 4px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 700;">
<span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:{'#10B981' if _copilot_active else '#F59E0B'};"></span>
{'AI Copilot Active' if _copilot_active else 'AI Copilot — Fallback Mode'}
</div>
</div>
</div>
<div style="text-align: right; display: flex; flex-direction: column; gap: 8px; align-items: flex-end;">
<div style="display: inline-flex; align-items: center; gap: 8px; background: {_status_bg}; color: {_status_fg}; border: 1px solid {_status_border}; padding: 6px 16px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">
<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {_status_dot}; box-shadow: 0 0 12px {'rgba(16,185,129,0.8)' if _last_call_ok is True else 'transparent'}; animation: pulse 2s infinite;"></span>
{_status_label}
</div>
<div style="display: inline-flex; align-items: center; gap: 8px; background: {_pipe_bg}; color: {_pipe_fg}; border: 1px solid {_pipe_border}; padding: 6px 16px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">
<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {_pipe_dot}; box-shadow: 0 0 12px {'rgba(59,130,246,0.5)' if _pipeline_live else 'transparent'};"></span>
{_pipe_label}
</div>
</div>
</div>
</div>
        """,
        unsafe_allow_html=True
    )

    # ── EMPTY STATE: HEADING, CAPABILITY CARDS, TOOL CHIPS ───────────────────
    if not st.session_state["copilot_messages"]:
        st.markdown(
            """
            <div style="margin-top: 8px; margin-bottom: 24px; text-align: center;">
                <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">How can I help you optimize your process today?</h3>
                <p style="color: #64748B; font-size: 0.95rem;">Ask a question below, or try one of these suggested causal capabilities.</p>
            </div>
            """, unsafe_allow_html=True
        )

        def set_question(question, key):
            st.session_state["copilot_pending_question"] = question
            st.session_state["copilot_pending_chip_key"] = key

        # (title, description, tags, icon, color, question, chip_key)
        cards = [
            ("What is the top bottleneck?", "Identify the most impactful constraint",
             ["Bottleneck Analysis", "Root Cause"], "⚠️", "warn",
             "What is the top bottleneck?", "bottleneck"),
            ("Show Causal KPI Impact", "See which factors drive your KPIs",
             ["Causal Impact", "KPI Analysis"], "📊", "purple",
             "Show Causal KPI Impact", "impact"),
            ("Run What-If Simulation", "Test scenarios & predict outcomes",
             ["Simulation", "Forecasting"], "🧪", "blue",
             "Run What-If Simulation", "impact"),
            ("Generate Executive Summary", "AI-powered insights & recommendations",
             ["AI Summary", "Insights"], "📋", "success",
             "Generate Executive Summary", "executive"),
        ]

        # st.container(border=True) is a real Streamlit primitive that
        # properly wraps both markdown AND a real st.button in one native
        # bordered box — sidesteps the raw-HTML-div-splitting bug above
        # entirely (a div with its OWN visible border, split across separate
        # st.markdown() calls, rendered as a stray empty colored box the
        # first time this was tried, since Streamlit closes each call's raw
        # HTML independently rather than truly nesting across calls).
        card_cols = [st.columns(2), st.columns(2)]
        for idx, (title, desc, tags, icon, color, question, key) in enumerate(cards):
            col = card_cols[idx // 2][idx % 2]
            _bg = {"warn": "#FEF3C7", "purple": "#EDE9FE", "blue": "#DBEAFE", "success": "#D1FAE5"}[color]
            _fg = {"warn": "#B45309", "purple": "#6D28D9", "blue": "#1D4ED8", "success": "#047857"}[color]
            _accent = {"warn": "#D97706", "purple": "#8B5CF6", "blue": "#3B82F6", "success": "#059669"}[color]
            with col:
                with st.container(border=True):
                    st.markdown(f'<div class="cop-card-accent-bar" style="background:{_accent};"></div>', unsafe_allow_html=True)
                    text_col, arrow_col = st.columns([6, 1])
                    with text_col:
                        _tag_html = "".join(
                            f'<span class="cop-tag" style="background:{_bg};color:{_fg};">{t}</span>' for t in tags
                        )
                        st.markdown(
                            f'<div class="cop-card-row">'
                            f'<div class="cop-card-icon" style="background:{_bg};">{icon}</div>'
                            f'<div style="flex:1;"><div class="cop-card-title">{title}</div>'
                            f'<div class="cop-card-desc">{desc}</div>'
                            f'<div class="cop-card-tags">{_tag_html}</div></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with arrow_col:
                        st.markdown(f'<div class="cop-card-arrow cop-card-arrow--{color}">', unsafe_allow_html=True)
                        if st.button("→", key=f"card_{key}_{idx}"):
                            set_question(question, key)
                        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

        # Second-tier tool row — every chip maps to a real backend intent key
        # (see copilot.py QUICK_CHIPS / FOLLOW_UP_POOL), not a decorative
        # dead button. "More Tools" opens the two remaining quick actions
        # that don't have room in the main row rather than doing nothing.
        # Labels are single words: the wrapper-div trick used elsewhere to
        # scope custom CSS (nowrap, smaller font) to just these six buttons
        # doesn't work on this Streamlit version — confirmed via DOM
        # inspection, it renders as an empty div, same bug as the capability
        # cards — and there's no key-based class hook either, so keeping
        # labels short enough to fit one line at the default button width
        # was the reliable fix rather than another CSS scoping attempt.
        tool_chips = [
            ("🔍 Discovery",   "Explain the discovered causal chain in this process", "chain"),
            ("🛡️ Conformance", "Check this process for conformance issues and deviations", "custom"),
            ("📈 Trends",      "Why are delays increasing over time?", "delays"),
            ("🎯 Drill-Down",  "What is the top bottleneck limiting performance?", "bottleneck"),
            ("⚖️ Compare",     "Compare suppliers and their impact on outcomes", "suppliers"),
        ]
        tool_cols = st.columns(6)
        for i, (label, question, key) in enumerate(tool_chips):
            with tool_cols[i]:
                if st.button(label, key=f"tool_{key}_{i}", width='stretch'):
                    set_question(question, key)
        with tool_cols[5]:
            with st.popover("⊞ More Tools", width='stretch'):
                if st.button("💡 Best intervention?", key="more_intervention", width='stretch'):
                    set_question("Best intervention?", "intervention")
                    st.rerun()
                if st.button("💰 ROI opportunities?", key="more_roi", width='stretch'):
                    set_question("What are the ROI opportunities?", "roi")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # ── CHAT INPUT LOGIC ──────────────────────────────────────────────────────
    pending_q = st.session_state.pop("copilot_pending_question", None)
    pending_chip_key = st.session_state.pop("copilot_pending_chip_key", None)

    # Quick prompts shown just above the (sticky) chat input — Streamlit pins
    # chat_input to the bottom of the viewport regardless of code order, so
    # placing these after chat_input would render them below the fold, not
    # visually below the bar; putting them here keeps them reachable and in
    # the same visual neighborhood as the reference design's bottom row.
    if not st.session_state["copilot_messages"]:
        bottom_chips = [
            ("✨ Analyze payment delays", "Analyze payment delays", "delays"),
            ("✨ Why is order processing slow?", "Why is order processing slow?", "bottleneck"),
            ("✨ Show impact of resource constraints", "Show impact of resource constraints", "impact"),
        ]
        bc_cols = st.columns(len(bottom_chips))
        for i, (label, question, key) in enumerate(bottom_chips):
            with bc_cols[i]:
                st.markdown('<div class="cop-bottom-chip">', unsafe_allow_html=True)
                if st.button(label, key=f"bottom_{key}_{i}", width='stretch'):
                    st.session_state["copilot_pending_question"] = question
                    st.session_state["copilot_pending_chip_key"] = key
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    user_q = st.chat_input("Ask Copilot to analyze root causes, simulate ROI, or explain causal graphs...")
    active_question = user_q or pending_q
    # A chip key only applies when the suggestion chip supplied the question —
    # free-typed text always falls back to keyword detection in call_cerebras.
    active_chip_key = None if user_q else pending_chip_key

    if active_question:
        # Save user msg (we don't need st.rerun() because chat_input already triggered a rerun)
        st.session_state["copilot_messages"].append({"role": "user", "content": active_question})

    # ── CHAT HISTORY RENDERING ────────────────────────────────────────────────
    for msg in st.session_state["copilot_messages"]:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            # Follow ups inline
            if msg["role"] == "assistant" and msg.get("follow_ups") and msg == st.session_state["copilot_messages"][-1]:
                st.markdown("<hr style='margin-top: 24px; margin-bottom: 16px; border: none; border-top: 1px solid #F1F5F9;'/>", unsafe_allow_html=True)
                fu_cols = st.columns(len(msg["follow_ups"]))
                for idx, fu in enumerate(msg["follow_ups"]):
                    with fu_cols[idx]:
                        st.markdown('<div class="saas-suggestion-btn saas-suggestion-btn--sm">', unsafe_allow_html=True)
                        if st.button("↳ " + fu, key=f"fu_{idx}_{len(st.session_state['copilot_messages'])}", width='stretch'):
                            st.session_state["copilot_pending_question"] = fu
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

    # ── API CALL & STREAMING ──────────────────────────────────────────────────
    # Check if last message is user, if so, fetch response
    if st.session_state["copilot_messages"] and st.session_state["copilot_messages"][-1]["role"] == "user":
        if _COPILOT_AVAILABLE:
            with st.chat_message("assistant", avatar="🤖"):
                # Status steps reflect real work as it happens, not a scripted
                # delay — each label only appears once the step it names has
                # actually started/finished.
                with st.status("Building process context...", expanded=True) as status:
                    context = _copilot_build_context(
                        dag=dag, dag_metrics=dag_metrics, scm=scm,
                        coefs=coefs, cfg=cfg, domain=domain, df=df,
                        do_result=do_result
                    )
                    st.write("✓ Process context assembled.")
                    status.update(label="Querying Cerebras LLM...")

                    stream_gen, confidence, follow_ups, used_fallback = _copilot_call_cerebras(
                        question=st.session_state["copilot_messages"][-1]["content"],
                        context=context,
                        api_key=_active_key,
                        model="gemma-4-31b",
                        domain=domain,
                        stream=True,
                        chip_key=active_chip_key,
                    )
                    st.session_state["copilot_last_call_ok"] = not used_fallback
                    st.write("✓ Response received." if not used_fallback else "⚠ Falling back to cached answer.")
                    status.update(label="Analysis Complete", state="complete", expanded=False)

                full_response = st.write_stream(stream_gen)

            st.session_state["copilot_messages"].append({
                "role": "assistant",
                "content": full_response,
                "confidence": confidence,
                "follow_ups": follow_ups
            })
            st.rerun()
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.error("Copilot module is unavailable.")

# Execute the fragment
render_copilot()
