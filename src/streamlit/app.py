"""
app.py — Entry point
Run:  streamlit run app.py
"""

import streamlit as st

from css.theme   import inject_theme
from css.sidebar import render_sidebar
from css.tx_form import render_tx_form
from css.panel   import render_result_panel, render_empty_panel
from web     import safe_analyze

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title  = "Precision Dash — Fraud Detection",
    page_icon   = "🛡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Inject global CSS theme ─────────────────────────────────────────────────
inject_theme()

st.markdown("""
<style>
/* ENSURE LAYOUT KHÔNG BỊ ĐÈ */
.block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
active_page = render_sidebar()

# ── Top bar ──────────────────────────────────────────────────────────────────
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("""
    <div style="padding:18px 24px 0;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                     color:#4a5568;text-transform:uppercase;letter-spacing:.12em;">
            Transaction Analysis
        </span>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;
                    font-weight:700;color:#e8eaf0;letter-spacing:-0.01em;">
            FRAUD_DETECTION_ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_right:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    lookup_id = st.text_input(
        label="Lookup TXN",
        placeholder="🔍  Lookup TXN…",
        label_visibility="collapsed",
    )

st.markdown("<div style='padding:0 24px;'>", unsafe_allow_html=True)
st.markdown("---")

# ── Main layout: form (left) | results (right) ──────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    form_data = render_tx_form()

# ── State management ─────────────────────────────────────────────────────────
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_warning" not in st.session_state:
    st.session_state["last_warning"] = None

if form_data:
    with st.spinner("Running analysis engine…"):
        result, warning = safe_analyze(form_data)
    st.session_state["last_result"]  = result
    st.session_state["last_warning"] = warning

with right_col:
    if st.session_state["last_warning"]:
        st.warning(st.session_state["last_warning"], icon="⚠️")

    if st.session_state["last_result"]:
        render_result_panel(st.session_state["last_result"])
    else:
        render_empty_panel()

st.markdown("</div>", unsafe_allow_html=True)
