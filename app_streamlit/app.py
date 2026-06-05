import requests
import streamlit as st
from datetime import datetime
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="Credit Portfolio Risk Engine",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #071f3d;
        --navy-2: #0b315f;
        --blue: #1f77ff;
        --blue-2: #0b5ed7;
        --green: #22c55e;
        --orange: #f59e0b;
        --red: #ef4444;
        --text: #0f172a;
        --muted: #64748b;
        --card: rgba(255,255,255,0.92);
        --border: rgba(148,163,184,0.28);
    }

    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 45%, #fbfdff 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061a33 0%, #082849 52%, #061a33 100%);
        color: white;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .block-container {
        padding-top: 2.3rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }

    .hero {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.7rem;
    }

    .hero h1 {
        font-size: 2.65rem;
        line-height: 1.05;
        margin: 0;
        color: var(--text);
        letter-spacing: -0.04em;
        font-weight: 800;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 0.55rem;
    }

    .badge {
        display: inline-block;
        margin-left: 0.75rem;
        padding: 0.32rem 0.65rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #0b5ed7;
        font-size: 1rem;
        font-weight: 700;
        vertical-align: middle;
    }

    .api-card {
        min-width: 210px;
        padding: 1rem 1.25rem;
        border-radius: 14px;
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        text-align: left;
    }

    .api-dot {
        height: 12px;
        width: 12px;
        background: var(--green);
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.55rem;
        box-shadow: 0 0 0 6px rgba(34,197,94,0.12);
    }

    .section-card {
        padding: 1.35rem 1.5rem 1.5rem 1.5rem;
        border-radius: 18px;
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 1rem;
    }

    .mini-note {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(59,130,246,0.25);
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        color: #1e3a8a;
        margin-top: 1.7rem;
    }

    .mini-note strong {
        color: #0b5ed7;
    }

    div[data-testid="stNumberInput"] label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px !important;
    }

    .stButton > button {
        display: block;
        margin: 1.35rem auto 0 auto;
        border-radius: 12px;
        border: 0;
        padding: 0.8rem 2.1rem;
        background: linear-gradient(135deg, #2563eb 0%, #0b5ed7 100%);
        color: white;
        font-weight: 800;
        font-size: 1.05rem;
        box-shadow: 0 14px 28px rgba(37,99,235,0.28);
    }

    .stButton > button:hover {
        filter: brightness(0.98);
        transform: translateY(-1px);
    }

    .result-grid {
        display: grid;
        grid-template-columns: 1fr 0.8fr 1.15fr;
        gap: 1rem;
        margin-top: 1rem;
    }

    .result-card {
        min-height: 190px;
        padding: 1.35rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.94);
        border: 1px solid var(--border);
        box-shadow: 0 16px 44px rgba(15, 23, 42, 0.08);
    }

    .probability {
        font-size: 3rem;
        font-weight: 900;
        color: #1f77ff;
        line-height: 1;
        margin-top: 0.55rem;
    }

    .risk-pill {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.65rem 1.05rem;
        border-radius: 12px;
        font-size: 1.45rem;
        font-weight: 900;
    }

    .risk-low { color: #15803d; background: #dcfce7; border: 1px solid #86efac; }
    .risk-medium { color: #b45309; background: #fef3c7; border: 1px solid #fcd34d; }
    .risk-high { color: #b91c1c; background: #fee2e2; border: 1px solid #fca5a5; }

    .bar-wrap {
        margin-top: 1.2rem;
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0%, #fde047 45%, #f97316 70%, #ef4444 100%);
        position: relative;
    }

    .bar-marker {
        position: absolute;
        top: -8px;
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 13px solid #2563eb;
        transform: translateX(-8px);
    }

    .summary-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid #e2e8f0;
        padding: 0.38rem 0;
        color: #334155;
    }

    .summary-row:last-child {
        border-bottom: 0;
    }

    .summary-value {
        font-weight: 800;
        color: #0f172a;
        text-align: right;
    }

    .sidebar-title {
        font-size: 1.45rem;
        font-weight: 900;
        line-height: 1.15;
        margin: 0.6rem 0 2rem 0;
    }

    .sidebar-card {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        margin-top: 1.2rem;
    }

    @media (max-width: 1050px) {
        .hero { flex-direction: column; }
        .result-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🛡️ Credit Risk<br>Engine</div>", unsafe_allow_html=True)
    st.markdown("### 🏠 Risk Calculator")
    st.markdown("### ℹ️ About Model")
    st.markdown("### 〰️ API Status")
    st.markdown("### 📘 Documentation")
    st.markdown(
        """
        <div class='sidebar-card'>
            <strong>Model Info</strong><br><br>
            <div style='display:flex;justify-content:space-between;'><span>Model</span><b>XGBoost v0.1</b></div>
            <div style='display:flex;justify-content:space-between;'><span>Features</span><b>258</b></div>
            <div style='display:flex;justify-content:space-between;'><span>Version</span><b>0.1.0</b></div>
            <div style='display:flex;justify-content:space-between;'><span>API</span><b>🟢 Online</b></div>
        </div>
        <div class='sidebar-card'>
            <strong>Need Help?</strong><br>
            <small>Check the README or FastAPI Swagger docs.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Header
st.markdown(
    """
    <div class='hero'>
        <div>
            <h1>Credit Portfolio Risk Engine <span class='badge'>v0.1</span></h1>
            <p>XGBoost model served through a FastAPI REST API</p>
        </div>
        <div class='api-card'>
            <div><span class='api-dot'></span><strong style='color:#15803d;'>API Connected</strong></div>
            <div style='color:#64748b;margin-top:0.25rem;'>localhost:8000</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>👥 Applicant / Portfolio Risk Inputs</div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    avg_dpd = st.number_input("Average DPD to closure last 24 months", value=0.0)
    init_transaction_code = st.number_input("Initial transaction code", value=1, step=1)
    num_rejects_9m = st.number_input("Number of rejects in last 9 months", value=0.0)

with right:
    bureau_a2_overdue_mean = st.number_input("Bureau A2 payment overdue mean", value=0.0)
    bureau_a1_overdue_sum = st.number_input("Bureau A1 max overdue amount sum", value=0.0)
    st.markdown(
        """
        <div class='mini-note'>
            <strong>About the inputs</strong><br>
            These are selected derived risk features. In production, most features would be generated automatically from internal systems and bureau data.
        </div>
        """,
        unsafe_allow_html=True,
    )

payload = {
    "features": {
        "avgdpdtolclosure24_3658938P": avg_dpd,
        "inittransactioncode_186L": int(init_transaction_code),
        "numrejects9m_859L": num_rejects_9m,
        "bureau_a2_light_pmts_overdue_1140A_mean": bureau_a2_overdue_mean,
        "bureau_a1_light_overdueamountmax_155A_sum": bureau_a1_overdue_sum,
    }
}

clicked = st.button("📊 Calculate Risk  ›")
st.markdown("</div>", unsafe_allow_html=True)

if clicked:
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        probability = float(result["default_probability"])
        risk_band = str(result["risk_band"]).lower()
        marker_left = min(max(probability * 100, 0), 100)

        risk_class = {
            "low": "risk-low",
            "medium": "risk-medium",
            "high": "risk-high",
        }.get(risk_band, "risk-medium")

        st.markdown(
            f"""
            <div class='result-grid'>
                <div class='result-card'>
                    <strong>Default Probability</strong>
                    <div class='probability'>{probability:.2%}</div>
                    <div style='color:#64748b;margin-top:0.25rem;'>Probability of Default</div>
                    <div class='bar-wrap'>
                        <div class='bar-marker' style='left:{marker_left}%;'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;color:#64748b;margin-top:0.55rem;'>
                        <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                    </div>
                </div>
                <div class='result-card'>
                    <strong>Risk Band</strong><br>
                    <div class='risk-pill {risk_class}'>🛡️ {risk_band.upper()}</div>
                    <p style='color:#475569;margin-top:1rem;'>The applicant / portfolio record has a <strong>{risk_band}</strong> level of risk.</p>
                </div>
                <div class='result-card'>
                    <strong>📈 Prediction Summary</strong><br><br>
                    <div class='summary-row'><span>Probability of Default</span><span class='summary-value'>{probability:.2%}</span></div>
                    <div class='summary-row'><span>Risk Band</span><span class='summary-value'>{risk_band.upper()}</span></div>
                    <div class='summary-row'><span>Features Provided</span><span class='summary-value'>{result.get('features_received')}</span></div>
                    <div class='summary-row'><span>Model Features</span><span class='summary-value'>{result.get('model_features')}</span></div>
                    <div class='summary-row'><span>Model</span><span class='summary-value'>XGBoost v0.1</span></div>
                    <div class='summary-row'><span>Prediction Time</span><span class='summary-value'>{datetime.now().strftime('%d %b %Y %H:%M:%S')}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except requests.exceptions.RequestException as error:
        st.error("Could not connect to FastAPI service. Make sure the API is running on localhost:8000.")
        st.code(str(error))
else:
    st.markdown(
        """
        <div class='result-grid'>
            <div class='result-card'>
                <strong>Default Probability</strong>
                <div class='probability'>—</div>
                <div style='color:#64748b;'>Run a prediction to display probability.</div>
            </div>
            <div class='result-card'>
                <strong>Risk Band</strong><br>
                <div class='risk-pill risk-medium'>WAITING</div>
            </div>
            <div class='result-card'>
                <strong>📈 Prediction Summary</strong><br><br>
                <div style='color:#64748b;'>No prediction yet.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
