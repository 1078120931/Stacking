import os
import uuid
import csv
from io import StringIO
from datetime import datetime

import numpy as np
import joblib
from PIL import Image
import streamlit as st

# =========================
# Fixed risk thresholds (percent)
# =========================
LOW_RISK_THRESHOLD = 10.0   # <10% = low risk
HIGH_RISK_THRESHOLD = 50.0  # 10–50% = intermediate, ≥50% = high

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="IPN Hemorrhage Risk — Xiangya Hospital",
    layout="wide",
    page_icon="🩸",
)

# -------------------------
# Session-level ID
# -------------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()

session_id = st.session_state["session_id"]

# =========================
# CSS Styling
# =========================
st.markdown(
    """
    <style>
        .main {
            padding: 0rem 3rem 3rem 3rem;
        }
        /* Xiangya header bar */
        .top-bar {
            background: linear-gradient(90deg, #0b7285, #1971c2);
            padding: 1.0rem 1.6rem;
            border-radius: 0 0 1.1rem 1.1rem;
            margin: -1.2rem -3rem 1.8rem -3rem;
            color: #ffffff;
        }
        .top-bar-title {
            font-size: 1.38rem;
            font-weight: 700;
        }
        .top-bar-subtitle {
            font-size: 0.9rem;
            opacity: 0.92;
        }
        .top-bar-right {
            font-size: 0.8rem;
            text-align: right;
        }
        /* Risk cards */
        .risk-card {
            border-radius: 0.9rem;
            padding: 1rem 1.4rem;
            margin-bottom: 0.9rem;
            font-size: 0.95rem;
        }
        .risk-low {
            background: #e8f5e9;
            border-left: 6px solid #43a047;
        }
        .risk-medium {
            background: #fff8e1;
            border-left: 6px solid #ffa000;
        }
        .risk-high {
            background: #ffebee;
            border-left: 6px solid #e53935;
        }
        .pill-label {
            display: inline-block;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 0.35rem;
        }
        .pill-low { background:#e8f5e9; color:#2e7d32; }
        .pill-medium { background:#fff8e1; color:#f9a825; }
        .pill-high { background:#ffebee; color:#c62828; }
        .small-muted {
            color: #777777;
            font-size: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Top Bar — Xiangya Hospital Header
# =========================
header_html = f"""
<div class="top-bar">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div class="top-bar-title">Xiangya Hospital · IPN Hemorrhage Decision Support System</div>
      <div class="top-bar-subtitle">
        Stacking Ensemble · Clinical Risk Prediction · Infected Pancreatic Necrosis (IPN)
      </div>
    </div>
    <div class="top-bar-right">
      <div>User: <b>Guest</b></div>
      <div>Session ID: {session_id}</div>
      <div>{datetime.now().strftime("%Y-%m-%d")}</div>
    </div>
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# =========================
# Page introduction
# =========================
st.title("🩸 Prediction of Intra-Abdominal Hemorrhage in Infected Pancreatic Necrosis (IPN)")

st.markdown(
    """
    This tool uses a **stacking machine learning model** to estimate the risk of  
    **clinically significant intra-abdominal hemorrhage** in patients with  
    **infected pancreatic necrosis (IPN)**.

    Enter the patient characteristics in the sidebar and click  
    **Predict hemorrhage risk** to obtain an individualized estimate and SHAP explanations.
    """
)

# =========================
# Sidebar Input
# =========================
with st.sidebar:
    st.header("Patient Features (IPN)")

    OF_num = st.selectbox("Organ failure (0 / 1 / 2)", [0,1,2])
    pancreatic_fis = st.selectbox("Postoperative pancreatic fistula (0=No,1=Yes)", [0,1])
    pan_MDRO = st.selectbox("Pus MDRO infection (0=No,1=Yes)", [0,1])
    blood_inf = st.selectbox("Bloodstream infection (0=No,1=Yes)", [0,1])
    age = st.number_input("Age (years)", 0,120,60)
    OF_time = st.number_input("Duration of organ failure (days)", 0,365,0)
    time_sur = st.number_input("Onset-to-intervention interval (days)", 0,365,0)

    st.markdown("---")
    predict_btn = st.button("▶ Predict hemorrhage risk", use_container_width=True)
    reset_btn = st.button("⟲ Reset session", use_container_width=True)

if reset_btn:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()
    st.experimental_rerun()

# =========================
# Load model
# =========================
@st.cache_resource(show_spinner="Loading model…")
def load_model(path="best_model_stack.pkl"):
    return joblib.load(path)

def load_image(path):
    return Image.open(path) if os.path.exists(path) else None

def build_export_csv(record: dict) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(record.keys()))
    writer.writeheader(); writer.writerow(record)
    return buffer.getvalue()

# =========================
# Main Layout
# =========================
col_left, col_right = st.columns([1.2,1])

with col_left:
    st.subheader("Prediction Result")

    prediction_made = False
    export_data = None

    if predict_btn:

        model = load_model()
        X = np.array([[OF_num, pancreatic_fis, pan_MDRO, blood_inf,
                       age, OF_time, time_sur]])

        prob = float(model.predict_proba(X)[0][1])
        prob = max(0.0, min(prob, 1.0))
        pct = prob * 100

        # Fixed risk thresholds
        low_thr = LOW_RISK_THRESHOLD
        high_thr = HIGH_RISK_THRESHOLD

        if pct < low_thr:
            risk_cat = "Low"
            css_class = "risk-low"
            pill_class = "pill-low"
            risk_msg = (
                "Low estimated risk of clinically significant intra-abdominal hemorrhage in IPN."
            )
        elif pct < high_thr:
            risk_cat = "Intermediate"
            css_class = "risk-medium"
            pill_class = "pill-medium"
            risk_msg = (
                "Intermediate hemorrhage risk. Close clinical monitoring is recommended."
            )
        else:
            risk_cat = "High"
            css_class = "risk-high"
            pill_class = "pill-high"
            risk_msg = (
                "High risk of intra-abdominal hemorrhage. Consider early imaging, vascular evaluation, "
                "and multidisciplinary intervention."
            )

        st.markdown(
            f"""
            <div class="risk-card {css_class}">
                <h4 style="margin-top:0;">Predicted hemorrhage risk: {pct:.1f}% 
                    <span class="pill-label {pill_class}">{risk_cat} risk</span>
                </h4>
                <p style="margin-bottom:0;">{risk_msg}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(prob)

        prediction_made = True

        export_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hemorrhage_probability": prob,
            "hemorrhage_risk_percent": pct,
            "risk_category": risk_cat,
            "OF_num": OF_num,
            "pancreatic_fistula": pancreatic_fis,
            "MDRO_infection": pan_MDRO,
            "bloodstream_infection": blood_inf,
            "age": age,
            "OF_duration_days": OF_time,
            "onset_to_intervention_days": time_sur,
        }

    else:
        st.info("Enter features in the sidebar and click **Predict hemorrhage risk**.")

    if prediction_made:
        csv_str = build_export_csv(export_data)
        st.download_button(
            "💾 Download prediction (CSV)",
            csv_str,
            file_name=f"IPN_hemorrhage_{session_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

with col_right:
    st.subheader("Clinical Interpretation (IPN Hemorrhage)")
    st.markdown(
        """
        Patients with **infected pancreatic necrosis (IPN)** are at risk of  
        **clinically significant intra-abdominal hemorrhage**, often due to:

        - inflammatory erosion of major vessels  
        - pseudoaneurysm formation  
        - infected necrotic tissue invading vascular structures  

        The stacking model integrates multiple predictors to estimate hemorrhage risk,  
        assisting clinicians in early surveillance and decision-making.
        """
    )

# =========================
# SHAP Visualisation
# =========================
st.markdown("---")
st.header("🔍 SHAP Model Explanation")

tab1, tab2 = st.tabs(["Base learners", "Stacking model"])

with tab1:
    img1 = load_image("summary_plot.png")
    st.image(img1, caption="SHAP values — base learners", use_column_width=True) if img1 else st.warning("summary_plot.png not found.")

with tab2:
    img2 = load_image("overall_shap.png")
    st.image(img2, caption="SHAP summary — stacking model", use_column_width=True) if img2 else st.warning("overall_shap.png not found.")

st.markdown("---")
st.caption("© 2025 Xiangya Hospital · IPN Hemorrhage Prediction System · Stacking Model Prototype")
