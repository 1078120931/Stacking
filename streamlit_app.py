# ============================================
# Imports
# ============================================

import os
import uuid
import csv
from io import StringIO
from datetime import datetime

import numpy as np
import joblib
from PIL import Image
import streamlit as st

# PDF generator (lightweight & Streamlit Cloud friendly)
from fpdf import FPDF


# ============================================
# Basic page configuration
# ============================================

st.set_page_config(
    page_title="IPN Hemorrhage Risk — Xiangya Hospital",
    layout="wide",
    page_icon="🩸",
)

# Generate session ID
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()
session_id = st.session_state["session_id"]


# ============================================
# CSS Styling (EASY-APP style)
# ============================================

st.markdown(
    """
    <style>
        .main {
            padding: 0rem 3rem 3rem 3rem;
        }

        .top-bar {
            background: linear-gradient(90deg, #0b7285, #1971c2);
            padding: 1.0rem 1.6rem;
            border-radius: 0 0 1.1rem 1.1rem;
            margin: -1.2rem -3rem 1.8rem -3rem;
            color: #ffffff;
        }
        .top-bar-title {
            font-size: 1.40rem;
            font-weight: 700;
        }
        .top-bar-subtitle {
            font-size: 0.90rem;
            opacity: 0.92;
        }
        .top-bar-right {
            font-size: 0.80rem;
            text-align: right;
        }

        .risk-card {
            border-radius: 0.9rem;
            padding: 1rem 1.4rem;
            margin-bottom: 0.9rem;
            font-size: 0.95rem;
        }
        .risk-low    { background: #e8f5e9; border-left: 6px solid #43a047; }
        .risk-medium { background: #fff8e1; border-left: 6px solid #ffa000; }
        .risk-high   { background: #ffebee; border-left: 6px solid #e53935; }

        .pill-label {
            display: inline-block;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 0.35rem;
        }
        .pill-low    { background:#e8f5e9; color:#2e7d32; }
        .pill-medium { background:#fff8e1; color:#f9a825; }
        .pill-high   { background:#ffebee; color:#c62828; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# Header bar
# ============================================

header_html = f"""
<div class="top-bar">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div class="top-bar-title">IPN Hemorrhage Decision Support System</div>
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


# ============================================
# Page introduction
# ============================================

st.title("🩸 Prediction of Intra-Abdominal Hemorrhage in Infected Pancreatic Necrosis (IPN)")

st.markdown(
    """
    This application uses a **stacking machine learning model** to estimate the risk of  
    **clinically significant intra-abdominal hemorrhage** in patients with  
    **infected pancreatic necrosis (IPN)**.

    Enter patient data in the sidebar and click  
    **Predict hemorrhage risk** to obtain an individualized estimate and SHAP explanations.
    """
)


# ============================================
# Sidebar Input
# ============================================

with st.sidebar:
    st.header("IPN Patient Features")

    OF_num = st.selectbox("Organ failure (0 / 1 / 2)", [0, 1, 2])
    pancreatic_fis = st.selectbox("Pancreatic fistula (0=No,1=Yes)", [0, 1])
    pan_MDRO = st.selectbox("Pus MDRO infection (0=No,1=Yes)", [0, 1])
    blood_inf = st.selectbox("Bloodstream infection (0=No,1=Yes)", [0, 1])

    age = st.number_input("Age (years)", 0, 120, 60)
    OF_time = st.number_input("Duration of organ failure (days)", 0, 365, 0)
    time_sur = st.number_input("Onset-to-intervention interval (days)", 0, 365, 0)

    st.markdown("---")
    predict_btn = st.button("▶ Predict hemorrhage risk", use_container_width=True)
    reset_btn   = st.button("⟲ Reset session", use_container_width=True)

if reset_btn:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()
    st.experimental_rerun()


# ============================================
# Load model & utilities
# ============================================

@st.cache_resource(show_spinner="Loading model…")
def load_model(path="best_model_stack.pkl"):
    return joblib.load(path)


def load_image(path):
    return Image.open(path) if os.path.exists(path) else None


# ============================================
# PDF Generation (FPDF)
# ============================================

def generate_pdf(data: dict):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Xiangya Hospital", ln=True)
    pdf.cell(0, 10, "IPN Hemorrhage Risk Report", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)

    for key, value in data.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)

    pdf_output = pdf.output(dest="S").encode("latin1")
    return pdf_output


# ============================================
# Main layout
# ============================================

col_left, col_right = st.columns([1.2, 1])


# ============================================
# Prediction
# ============================================

with col_left:
    st.subheader("Prediction Result")

    if predict_btn:

        model = load_model()
        X = np.array([[OF_num, pancreatic_fis, pan_MDRO, blood_inf, age, OF_time, time_sur]])

        prob = float(model.predict_proba(X)[0][1])
        prob = max(0.0, min(prob, 1.0))
        pct = prob * 100

        if pct < 10:
            risk_cat = "Low"
            css_class = "risk-low"
            pill_class = "pill-low"
            risk_msg = "Low estimated risk of intra-abdominal hemorrhage in IPN."
        elif pct < 50:
            risk_cat = "Intermediate"
            css_class = "risk-medium"
            pill_class = "pill-medium"
            risk_msg = "Intermediate risk of intra-abdominal hemorrhage. Monitoring recommended."
        else:
            risk_cat = "High"
            css_class = "risk-high"
            pill_class = "pill-high"
            risk_msg = (
                "High risk of intra-abdominal hemorrhage. Consider CTA, vascular evaluation, "
                "and timely intervention."
            )

        st.markdown(
            f"""
            <div class="risk-card {css_class}">
                <h4>Predicted hemorrhage risk: {pct:.1f}% 
                    <span class="pill-label {pill_class}">{risk_cat} risk</span>
                </h4>
                <p>{risk_msg}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(prob)

        # Prepare data for PDF report
        report_data = {
            "Session ID": session_id,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Risk (%)": f"{pct:.1f}",
            "Risk category": risk_cat,
            "Organ failure": OF_num,
            "Pancreatic fistula": pancreatic_fis,
            "Pus MDRO infection": pan_MDRO,
            "Bloodstream infection": blood_inf,
            "Age": age,
            "OF duration (days)": OF_time,
            "Onset-to-intervention (days)": time_sur
        }

        pdf_bytes = generate_pdf(report_data)

        st.download_button(
            "🧾 Download PDF Report",
            pdf_bytes,
            file_name=f"IPN_hemorrhage_report_{session_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ============================================
# Right side: Model explanation
# ============================================

with col_right:
    st.subheader("Model Overview")

    st.markdown(
        """
        **Outcome**  
        Probability of **intra-abdominal hemorrhage** in patients with **infected pancreatic necrosis (IPN)**.

        **Predictors included**  
        - Organ failure (0/1/2)  
        - Pancreatic fistula  
        - Pus MDRO infection  
        - Bloodstream infection  
        - Age  
        - Duration of organ failure  
        - Onset-to-intervention interval  
        """
    )


# ============================================
# SHAP Visualisation
# ============================================

st.markdown("---")
st.header("🔍 SHAP-Based Model Explanation")

tab1, tab2 = st.tabs(["Base learners", "Stacking model"])

with tab1:
    img1 = load_image("summary_plot.png")
    if img1:
        st.image(img1, caption="SHAP — Base learners", use_column_width=True)
    else:
        st.warning("summary_plot.png not found.")

with tab2:
    img2 = load_image("overall_shap.png")
    if img2:
        st.image(img2, caption="SHAP — Final stacking model", use_column_width=True)
    else:
        st.warning("overall_shap.png not found.")


st.markdown("---")
st.caption("© 2025 Xiangya Hospital · IPN Intra-Abdominal Hemorrhage Prediction System")
