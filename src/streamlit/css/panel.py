import re
import streamlit as st


PANEL_CSS = """
<style>
.risk-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a5568;
    margin-bottom: 12px;
}
.fraud-detected-box {
    background: rgba(255,71,87,0.08);
    border: 1px solid rgba(255,71,87,0.25);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.fraud-detected-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #ff4757;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.confidence-badge {
    background: rgba(0,229,195,0.12);
    border: 1px solid rgba(0,229,195,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    min-width: 72px;
}
.confidence-label { font-size: 9px; font-family:'JetBrains Mono',monospace; color:#4a5568; text-transform:uppercase; }
.confidence-value { font-size: 22px; font-weight: 800; color: #00e5c3; font-family:'JetBrains Mono',monospace; }

.safe-box {
    background: rgba(0,229,195,0.06);
    border: 1px solid rgba(0,229,195,0.2);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.safe-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #00e5c3;
    line-height: 1.1;
}
.severity-row {
    display: flex;
    align-items: center;
    gap: 20px;
    margin: 16px 0;
}
.severity-ring {
    width: 72px; height: 72px;
    border-radius: 50%;
    border: 4px solid;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    flex-shrink: 0;
}
.severity-ring.HIGH   { border-color: #ff4757; box-shadow: 0 0 16px rgba(255,71,87,0.3); }
.severity-ring.MEDIUM { border-color: #ffa502; box-shadow: 0 0 16px rgba(255,165,2,0.3); }
.severity-ring.LOW    { border-color: #00e5c3; box-shadow: 0 0 16px rgba(0,229,195,0.3); }
.severity-ring-label { font-size: 8px; color:#8892aa; font-family:'JetBrains Mono',monospace; }
.severity-ring-value { font-size: 11px; font-weight:700; font-family:'JetBrains Mono',monospace; }
.severity-ring.HIGH   .severity-ring-value { color: #ff4757; }
.severity-ring.MEDIUM .severity-ring-value { color: #ffa502; }
.severity-ring.LOW    .severity-ring-value { color: #00e5c3; }

.factors-title {
    font-family:'JetBrains Mono',monospace;
    font-size:10px; text-transform:uppercase; letter-spacing:0.1em;
    color:#4a5568; margin: 16px 0 8px;
}
.factor-item {
    display: flex; align-items: flex-start; gap: 10px;
    background: #1a1f2e; border: 1px solid #2a3045;
    border-radius: 8px; padding: 10px 12px; margin: 6px 0;
}
.factor-dot { width:8px; height:8px; border-radius:50%; margin-top:4px; flex-shrink:0; }
.factor-name { font-size:13px; font-weight:600; color:#e8eaf0; font-family:'Space Grotesk',sans-serif; }
.factor-detail { font-size:10px; color:#4a5568; font-family:'JetBrains Mono',monospace; margin-top:2px; }

.result-footer { display:flex; gap:8px; margin-top:16px; }
.btn-view, .btn-freeze {
    flex:1; padding:9px; border-radius:7px; text-align:center;
    font-family:'JetBrains Mono',monospace; font-size:11px;
    font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
    background:#1a1f2e; border:1px solid #2a3045; color:#8892aa;
}
.btn-report {
    flex:1; padding:9px; border-radius:7px; text-align:center;
    font-family:'JetBrains Mono',monospace; font-size:11px;
    font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
    background:#ff4757; border:none; color:#fff;
}

.meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }
.meta-card { background:#1a1f2e; border:1px solid #2a3045; border-radius:8px; padding:10px 12px; }
.meta-card-label { font-size:9px; font-family:'JetBrains Mono',monospace; color:#4a5568; text-transform:uppercase; }
.meta-card-value { font-size:13px; font-weight:600; color:#e8eaf0; margin-top:2px; word-break:break-all; }
.meta-card-sub   { font-size:10px; color:#4a5568; font-family:'JetBrains Mono',monospace; }

.source-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(74,158,255,0.1); border: 1px solid rgba(74,158,255,0.25);
    border-radius: 6px; padding: 4px 10px; margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #4a9eff;
}
.source-dot { width:6px; height:6px; border-radius:50%; background:#4a9eff; }
</style>
"""

_SEVERITY_COLOR = {"HIGH": "#ff4757", "MEDIUM": "#ffa502", "LOW": "#00e5c3"}
_FACTOR_COLORS  = ["#ff4757", "#ffa502", "#4a9eff"]


def _strip_html(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"</?[^>]+>", "", text)   # remove ALL tags (kể cả </div>)
    return text.replace("</div>", "").strip()  # double safety

def render_result_panel(result: dict):
    """
    Render the fraud analysis result panel.

    Reads from normalized keys (set by web._normalize):
      - is_fraud        bool
      - confidence      float  (mapped from fraud_score)
      - severity        str    (mapped from risk_level, uppercased)
      - risk_factors    list[{name, detail}]  (mapped from triggered_rules)
      - prediction_time str
    """
    st.markdown(PANEL_CSS, unsafe_allow_html=True)

    # ── Read normalized fields ─────────────────────────────────────────────
    is_fraud    = result.get("is_fraud", False)
    confidence  = result.get("confidence", 0.0)
    severity    = result.get("severity", "LOW").upper()
    factors     = result.get("risk_factors", [])
    pred_time   = result.get("prediction_time", "—")
    origin_node = result.get("origin_node", "—")
    latency     = result.get("latency", "—")

    sev_color = _SEVERITY_COLOR.get(severity, "#4a5568")
    conf_pct  = f"{confidence * 100:.1f}%"

    # Source badge
    st.markdown("""
    <div class="source-badge">
        <div class="source-dot"></div>
        LIVE · FastAPI Prediction
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="risk-label">RISK EVALUATION STATE</div>', unsafe_allow_html=True)

    # ── Fraud / Safe status box ────────────────────────────────────────────
    if is_fraud:
        st.markdown(f"""
        <div class="fraud-detected-box">
            <div class="fraud-detected-text">FRAUD<br>DETECTED</div>
            <div class="confidence-badge">
                <div class="confidence-label">FRAUD SCORE</div>
                <div class="confidence-value">{conf_pct}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe-box">
            <div class="safe-text">TRANSACTION<br>SAFE</div>
            <div class="confidence-badge">
                <div class="confidence-label">FRAUD SCORE</div>
                <div class="confidence-value">{conf_pct}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Severity ring ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="severity-row">
        <div class="severity-ring {severity}">
            <div class="severity-ring-label">SEVERITY</div>
            <div class="severity-ring-value">{severity}</div>
        </div>
        <div style="flex:1">
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                        color:#4a5568; text-transform:uppercase; margin-bottom:6px;">
                RISK LEVEL
            </div>
            <div style="font-size:15px; font-weight:700; color:{sev_color};">
                {severity} RISK
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ── Triggered rules ────────────────────────────────────────────────────
    if factors:
        st.markdown("**TRIGGERED RULES**")

        for i, factor in enumerate(factors[:5]):
            if isinstance(factor, dict):
                name = _strip_html(factor.get("name", ""))
            else:
                name = _strip_html(str(factor))

            if not name:
                continue

            color = _FACTOR_COLORS[i % len(_FACTOR_COLORS)]

            col1, col2 = st.columns([1, 20])

            with col1:
                st.markdown(
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{color};margin-top:6px;"></div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(f"**{name}**")

    # ── Footer action buttons ──────────────────────────────────────────────
    st.markdown("""
    <div class="result-footer">
        <div class="btn-view">VIEW CLUSTER</div>
        <div class="btn-freeze">FREEZE CARD</div>
        <div class="btn-report">REPORT FRAUD</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Meta info ──────────────────────────────────────────────────────────
    pred_display = pred_time
    if "T" in str(pred_time):
        try:
            pred_display = pred_time.replace("T", " ").split(".")[0]
        except Exception:
            pass

    st.markdown(f"""
    <div class="meta-grid">
        <div class="meta-card">
            <div class="meta-card-label">PREDICTION TIME</div>
            <div class="meta-card-value" style="font-size:11px;">{pred_display}</div>
        </div>
        <div class="meta-card">
            <div class="meta-card-label">ORIGIN NODE</div>
            <div class="meta-card-value">{origin_node}</div>
            <div class="meta-card-sub">Latency: {latency}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Raw API response expander ──────────────────────────────────────────
    raw = result.get("_raw")
    if raw:
        with st.expander("🔍  Raw API Response"):
            st.json(raw)


def render_empty_panel():
    """Placeholder shown before any analysis is run."""
    st.markdown(PANEL_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div style="height:400px; display:flex; align-items:center; justify-content:center;
                flex-direction:column; gap:12px; border:1px dashed #2a3045;
                border-radius:12px; margin-top:8px;">
        <div style="font-size:32px; opacity:0.3;">🔍</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                    color:#4a5568; text-transform:uppercase; letter-spacing:0.1em;">
            Awaiting Analysis
        </div>
        <div style="font-size:12px; color:#2a3045; font-family:'JetBrains Mono',monospace;">
            Fill form → Run Analysis Engine
        </div>
    </div>
    """, unsafe_allow_html=True)