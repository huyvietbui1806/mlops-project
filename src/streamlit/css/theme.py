DARK_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0d0f14;
    --bg-secondary: #131720;
    --bg-card: #1a1f2e;
    --bg-card-hover: #1f2538;
    --border: #2a3045;
    --border-accent: #3d4a6b;
    --text-primary: #e8eaf0;
    --text-secondary: #8892aa;
    --text-muted: #4a5568;
    --accent-teal: #00e5c3;
    --accent-teal-dim: rgba(0,229,195,0.12);
    --accent-red: #ff4757;
    --accent-red-dim: rgba(255,71,87,0.12);
    --accent-amber: #ffa502;
    --accent-amber-dim: rgba(255,165,2,0.12);
    --accent-blue: #4a9eff;
    --severity-high: #ff4757;
    --font-mono: 'JetBrains Mono', monospace;
    --font-sans: 'Space Grotesk', sans-serif;
    --radius: 8px;
    --radius-lg: 12px;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: var(--bg-primary) !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    width: 220px !important;
    min-width: 220px !important;
}
[data-testid="stSidebar"] .stMarkdown { padding: 0 !important; }

/* ── INPUTS ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 0 2px var(--accent-teal-dim) !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}

/* ── LABELS ── */
label, [data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: var(--font-mono) !important;
}

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; }

/* ── BUTTON ── */
[data-testid="stButton"] button {
    background: var(--accent-red) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: opacity 0.2s;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--accent-teal) !important; }

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
}

/* ── COLUMNS GAP ── */
[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
</style>
"""


def inject_theme():
    """Inject the dark theme CSS into the Streamlit app."""
    import streamlit as st
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
