import streamlit as st

NAV_ITEM_STYLE = """
<style>
/* Reduce sidebar width */
[data-testid="stSidebar"] {
    max-width: 200px !important;
}

[data-testid="stSidebar"] > div:first-child {
    width: 200px !important;
}

.sidebar-container { 
    padding-bottom: 100px; 
}

.nav-section { 
    padding: 12px 10px; 
    margin-top: 10px; 
}

.nav-section:first-of-type { 
    margin-top: 0; 
}

.nav-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a6b7d;
    padding: 6px 10px 4px;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 10px;
    border-radius: 6px;
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #8892aa;
    margin: 2px 0;
    transition: all 0.2s ease;
    border-left: 3px solid transparent;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.nav-item:hover {
    background: rgba(0, 229, 195, 0.08);
    color: #00e5c3;
    border-left-color: #00e5c3;
}

.nav-item.active {
    background: rgba(0, 229, 195, 0.12);
    color: #00e5c3;
    border-left-color: #00e5c3;
}

.nav-icon { 
    font-size: 14px;
    min-width: 16px;
    flex-shrink: 0;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px 10px 12px;
    border-bottom: 1px solid #2a3045;
    margin-bottom: 8px;
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 14px;
    color: #e8eaf0;
    letter-spacing: -0.01em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.brand-icon { 
    font-size: 20px;
    flex-shrink: 0;
}

.sidebar-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 200px;
    padding: 10px;
    border-top: 1px solid #2a3045;
    background: #131720;
}

.user-row {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}

.avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00e5c3, #4a9eff);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: #0d0f14;
    flex-shrink: 0;
}

.user-info { 
    flex: 1;
    min-width: 0;
}

.user-name { 
    font-size: 11px; 
    font-weight: 600; 
    color: #e8eaf0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.user-role { 
    font-size: 9px; 
    color: #5a6b7d; 
    font-family: 'JetBrains Mono', monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
"""

def render_sidebar() -> str:
    """Render compact fraud detection sidebar."""
    st.markdown(NAV_ITEM_STYLE, unsafe_allow_html=True)
    
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sidebar-brand">
            <div class="brand-icon">🛡️</div>
            <div class="brand-name">FraudGuard</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Navigation
        st.markdown('<div class="nav-label">Main</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="nav-item"><span class="nav-icon">📊</span> Dashboard</div>
        <div class="nav-item"><span class="nav-icon">💳</span> Transactions</div>
        <div class="nav-item"><span class="nav-icon">🚨</span> Alerts</div>
        """, unsafe_allow_html=True)
        
        # System
        st.markdown('<div class="nav-label">System</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="nav-item"><span class="nav-icon">🔧</span> Settings</div>
        <div class="nav-item"><span class="nav-icon">📚</span> Docs</div>
        <div class="nav-item"><span class="nav-icon">🔐</span> Security</div>
        """, unsafe_allow_html=True)
        
        # User Footer
        st.markdown("""
        <div class="sidebar-footer">
            <div class="user-row">
                <div class="avatar">KL</div>
                <div class="user-info">
                    <div class="user-name">Klinh</div>
                    <div class="user-role">ADMIN</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return "dashboard"