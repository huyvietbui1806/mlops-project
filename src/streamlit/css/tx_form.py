from __future__ import annotations

import streamlit as st


FORM_CSS = """
<style>
.form-header-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a5568;
    margin-bottom: 4px;
}
.form-header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.form-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4a5568;
    margin-bottom: 20px;
    line-height: 1.5;
}
.section-divider {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a5568;
    border-bottom: 1px solid #2a3045;
    padding-bottom: 6px;
    margin: 18px 0 12px;
}
.neural-status {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(0,229,195,0.06);
    border: 1px solid rgba(0,229,195,0.15);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 16px;
}
.neural-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00e5c3;
    animation: pulse 1.5s infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.4; transform:scale(0.7); }
}
.neural-text {
    font-size:12px;
    color:#8892aa;
    font-family:'Space Grotesk',sans-serif;
}
.neural-bold {
    font-weight:600;
    color:#00e5c3;
}
</style>
"""


MERCHANT_CATEGORIES = [
    "-- Select --",
    "grocery",
    "online_retail",
    "restaurant",
    "gas_station",
    "atm",
    "clothing",
    "pharmacy",
    "electronics",
    "hotel",
    "travel",
]

DEVICE_TYPES = [
    "-- Select --",
    "mobile_app",
    "pos_terminal",
    "web_browser",
    "atm",
    "phone_ivr",
]

DAY_OF_WEEK_OPTIONS = {
    "-- Select --": None,
    "0 — Monday": 0,
    "1 — Tuesday": 1,
    "2 — Wednesday": 2,
    "3 — Thursday": 3,
    "4 — Friday": 4,
    "5 — Saturday": 5,
    "6 — Sunday": 6,
}

BINARY_OPTIONS = {
    "-- Select --": None,
    "No (0)": 0,
    "Yes (1)": 1,
}


def _section(label: str) -> None:
    st.markdown(
        f'<div class="section-divider">{label}</div>',
        unsafe_allow_html=True,
    )


def _parse_text(label: str, value: str, errors: list[str]) -> str | None:
    value = str(value).strip()
    if not value:
        errors.append(f"{label} is required.")
        return None
    return value


def _parse_int(
    label: str,
    value: str,
    errors: list[str],
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    value = str(value).strip()

    if not value:
        errors.append(f"{label} is required.")
        return None

    try:
        parsed = int(value)
    except ValueError:
        errors.append(f"{label} must be an integer.")
        return None

    if min_value is not None and parsed < min_value:
        errors.append(f"{label} must be >= {min_value}.")
        return None

    if max_value is not None and parsed > max_value:
        errors.append(f"{label} must be <= {max_value}.")
        return None

    return parsed


def _parse_float(
    label: str,
    value: str,
    errors: list[str],
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    value = str(value).strip()

    if not value:
        errors.append(f"{label} is required.")
        return None

    try:
        parsed = float(value)
    except ValueError:
        errors.append(f"{label} must be a number.")
        return None

    if min_value is not None and parsed < min_value:
        errors.append(f"{label} must be >= {min_value}.")
        return None

    if max_value is not None and parsed > max_value:
        errors.append(f"{label} must be <= {max_value}.")
        return None

    return parsed


def _parse_choice(
    label: str,
    selected_label: str,
    mapping: dict[str, int | None],
    errors: list[str],
) -> int | None:
    value = mapping.get(selected_label)

    if value is None:
        errors.append(f"{label} is required.")
        return None

    return int(value)


def _parse_select_text(
    label: str,
    value: str,
    errors: list[str],
) -> str | None:
    if not value or value == "-- Select --":
        errors.append(f"{label} is required.")
        return None

    return value


def render_tx_form() -> dict | None:
    """
    Full transaction form.

    Users manually provide all fields required by FastAPI /predict.
    This UI does not run the model locally.
    It only sends the transaction payload to FastAPI and displays the API result.
    """
    st.markdown(FORM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="form-header-tag">TRANSACTION ANALYSIS</div>
        <div class="form-header-title">Transaction<br>Parameters</div>
        <div class="form-subtitle">
            Fill all transaction features below. This Streamlit UI sends the JSON payload
            to FastAPI <b>/predict</b> and displays the returned fraud_score and is_fraud.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("tx_form", clear_on_submit=False):
        _section("Identity")

        col1, col2 = st.columns(2)
        with col1:
            transaction_id_raw = st.text_input(
                "Transaction ID *",
                placeholder="TXN-000001",
            )
        with col2:
            user_id_raw = st.text_input(
                "User ID *",
                placeholder="USR-000001",
            )

        _section("Time Context")

        col1, col2, col3 = st.columns(3)
        with col1:
            hour_of_day_raw = st.text_input(
                "Hour of Day *",
                placeholder="0-23",
            )
        with col2:
            day_of_week_label = st.selectbox(
                "Day of Week *",
                options=list(DAY_OF_WEEK_OPTIONS.keys()),
            )
        with col3:
            is_weekend_label = st.selectbox(
                "Is Weekend *",
                options=list(BINARY_OPTIONS.keys()),
            )

        _section("Transaction Details")

        col1, col2 = st.columns(2)
        with col1:
            amount_raw = st.text_input(
                "Amount *",
                placeholder="100.00",
            )
        with col2:
            merchant_country_raw = st.text_input(
                "Merchant Country *",
                placeholder="US",
                max_chars=3,
            )

        col1, col2 = st.columns(2)
        with col1:
            merchant_category_raw = st.selectbox(
                "Merchant Category *",
                options=MERCHANT_CATEGORIES,
            )
        with col2:
            mcc_code_raw = st.text_input(
                "MCC Code *",
                placeholder="5045",
            )

        _section("Device & Security")

        col1, col2 = st.columns(2)
        with col1:
            device_type_raw = st.selectbox(
                "Device Type *",
                options=DEVICE_TYPES,
            )
        with col2:
            ip_risk_score_raw = st.text_input(
                "IP Risk Score *",
                placeholder="0.10",
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            card_present_label = st.selectbox(
                "Card Present *",
                options=list(BINARY_OPTIONS.keys()),
            )
        with col2:
            device_known_label = st.selectbox(
                "Device Known *",
                options=list(BINARY_OPTIONS.keys()),
            )
        with col3:
            is_foreign_txn_label = st.selectbox(
                "Foreign Txn *",
                options=list(BINARY_OPTIONS.keys()),
            )
        with col4:
            has_2fa_label = st.selectbox(
                "Has 2FA *",
                options=list(BINARY_OPTIONS.keys()),
            )

        _section("Behavioural Signals")

        col1, col2, col3 = st.columns(3)
        with col1:
            time_since_last_s_raw = st.text_input(
                "Time Since Last Txn (s) *",
                placeholder="3600",
            )
        with col2:
            velocity_1h_raw = st.text_input(
                "Velocity Last 1h *",
                placeholder="1",
            )
        with col3:
            amount_vs_avg_ratio_raw = st.text_input(
                "Amount vs Avg Ratio *",
                placeholder="1.00",
            )

        _section("Account Info")

        col1, col2 = st.columns(2)
        with col1:
            account_age_days_raw = st.text_input(
                "Account Age Days *",
                placeholder="365",
            )
        with col2:
            credit_limit_raw = st.text_input(
                "Credit Limit *",
                placeholder="5000.00",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "⚡  RUN ANALYSIS ENGINE",
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="neural-status">
            <div class="neural-dot"></div>
            <div class="neural-text">
                <span class="neural-bold">FastAPI Client Active</span> —
                This dashboard only collects transaction features and sends them to the
                deployed FastAPI model service for real-time fraud prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not submitted:
        return None

    errors: list[str] = []

    transaction_id = _parse_text("Transaction ID", transaction_id_raw, errors)
    user_id = _parse_text("User ID", user_id_raw, errors)

    hour_of_day = _parse_int(
        "Hour of Day",
        hour_of_day_raw,
        errors,
        min_value=0,
        max_value=23,
    )
    day_of_week = _parse_choice(
        "Day of Week",
        day_of_week_label,
        DAY_OF_WEEK_OPTIONS,
        errors,
    )
    is_weekend = _parse_choice(
        "Is Weekend",
        is_weekend_label,
        BINARY_OPTIONS,
        errors,
    )

    amount = _parse_float(
        "Amount",
        amount_raw,
        errors,
        min_value=0.0,
    )
    merchant_country = _parse_text(
        "Merchant Country",
        merchant_country_raw,
        errors,
    )
    merchant_category = _parse_select_text(
        "Merchant Category",
        merchant_category_raw,
        errors,
    )
    mcc_code = _parse_int(
        "MCC Code",
        mcc_code_raw,
        errors,
        min_value=0,
        max_value=9999,
    )

    device_type = _parse_select_text(
        "Device Type",
        device_type_raw,
        errors,
    )
    ip_risk_score = _parse_float(
        "IP Risk Score",
        ip_risk_score_raw,
        errors,
        min_value=0.0,
        max_value=9999,
    )

    card_present = _parse_choice(
        "Card Present",
        card_present_label,
        BINARY_OPTIONS,
        errors,
    )
    device_known = _parse_choice(
        "Device Known",
        device_known_label,
        BINARY_OPTIONS,
        errors,
    )
    is_foreign_txn = _parse_choice(
        "Foreign Txn",
        is_foreign_txn_label,
        BINARY_OPTIONS,
        errors,
    )
    has_2fa = _parse_choice(
        "Has 2FA",
        has_2fa_label,
        BINARY_OPTIONS,
        errors,
    )

    time_since_last_s = _parse_int(
        "Time Since Last Txn",
        time_since_last_s_raw,
        errors,
        min_value=0,
    )
    velocity_1h = _parse_int(
        "Velocity Last 1h",
        velocity_1h_raw,
        errors,
        min_value=0,
    )
    amount_vs_avg_ratio = _parse_float(
        "Amount vs Avg Ratio",
        amount_vs_avg_ratio_raw,
        errors,
        min_value=0.0,
    )

    account_age_days = _parse_int(
        "Account Age Days",
        account_age_days_raw,
        errors,
        min_value=0,
    )
    credit_limit = _parse_float(
        "Credit Limit",
        credit_limit_raw,
        errors,
        min_value=0.0,
    )

    if merchant_country is not None:
        merchant_country = merchant_country.strip().upper()

    if errors:
        for error in errors:
            st.error(f"⚠️  {error}")
        return None

    return {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "hour_of_day": int(hour_of_day),
        "day_of_week": int(day_of_week),
        "is_weekend": int(is_weekend),
        "amount": float(amount),
        "card_present": int(card_present),
        "device_known": int(device_known),
        "is_foreign_txn": int(is_foreign_txn),
        "has_2fa": int(has_2fa),
        "time_since_last_s": int(time_since_last_s),
        "velocity_1h": int(velocity_1h),
        "amount_vs_avg_ratio": float(amount_vs_avg_ratio),
        "account_age_days": int(account_age_days),
        "credit_limit": float(credit_limit),
        "merchant_category": merchant_category,
        "merchant_country": merchant_country,
        "device_type": device_type,
        "mcc_code": int(mcc_code),
        "ip_risk_score": float(ip_risk_score),
    }