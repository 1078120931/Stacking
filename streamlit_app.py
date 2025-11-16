# ============================================
# Imports
# ============================================

import os
import uuid
import csv
from io import StringIO, BytesIO
from datetime import datetime

import numpy as np
import joblib
from PIL import Image
import streamlit as st

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet


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
# Custom CSS Styling
# ============================================

st.markdown(
    """
    <style>
        .main {
            padding: 0rem 3rem 3rem 3rem;
        }

        /* Header bar */
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
        .pill-low    { background:#e8f5e9; color:#2e7d32; }
        .pill-medium { background:#fff8e1; color:#f9a825; }
        .pill-high   { background:#ffebee; color:#c62828; }

        .small-muted {
            color: #777777;
            font-size: 0.80rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# Header bar (EASY-APP style)
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
# Main title + introduction
# ============================================

st.title("🩸 Prediction of Intra-Abdominal Hemorrhage in Infected Pancreatic Necrosis (IPN)")

st.markdown(
    """
    This tool uses a **stacking machine learning model** to estimate the risk of  
    **clinically significant intra-abdominal hemorrhage** in patients with  
    **infected pancreatic necrosis (IPN)**.

    Enter patient characteristics in the sidebar and click  
    **Predict hemorrhage risk** to obtain an individualized estimate and SHAP-based explanations.
    """
)


# ============================================
# Sidebar Input
# ============================================

with st.sidebar:
    st.header("Patient Features (IPN)")

    OF_num = st.selectbox("Organ failure (0 / 1 / 2)", [0, 1, 2])
    pancreatic_fis = st.selectbox("Postoperative pancreatic fistula (0=No,1=Yes)", [0, 1])
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
# Generate PDF report
# ============================================

def generate_pdf(data: dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, 800, "Xiangya Hospital")
    c.drawString(30, 780, "IPN Hemorrhage Risk Report")

    c.setFont("Helvetica", 11)
    y = 745
    for key, value in data.items():
        c.drawString(30, y, f"{key}: {value}")
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ============================================
# Layout: left (prediction) | right (explanation)
# ============================================

col_left, col_right = st.columns([1.2, 1])


# ============================================
# Prediction Panel
# ============================================

with col_left:
    st.subheader("Prediction Result")

    if predict_btn:

        model = load_model()
        X = np.array([[OF_num, pancreatic_fis, pan_MDRO, blood_inf, age, OF_time, time_sur]])

        prob = model.predict_proba(X)[0][1]
        prob = float(max(0.0, min(prob, 1.0)))
        pct  = prob * 100

        if pct < 10:
            risk_cat   = "Low"
            css_class  = "risk-low"
            pill_class = "pill-low"
            risk_msg   = "Low estimated risk of clinically significant intra-abdominal hemorrhage in IPN."
        elif pct < 50:
            risk_cat   = "Intermediate"
            css_class  = "risk-medium"
            pill_class = "pill-medium"
            risk_msg   = "Intermediate hemorrhage risk. Close monitoring is recommended."
        else:
            risk_cat   = "High"
            css_class  = "risk-high"
            pill_class = "pill-high"
            risk_msg   = (
                "High risk of intra-abdominal hemorrhage. Consider early vascular evaluation, "
                "contrast-enhanced imaging, and timely intervention."
            )

        # Risk card
        st.markdown(
            f"""
            <div class="risk-card {css_class}">
                <h4 style="margin-top:0;">Predicted hemorrhage risk: {pct:.1f}% 
                    <span class="pill-label {pill_class}">{risk_cat} risk</span>
                </h4>
                <p>{risk_msg}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(prob)

        # PDF export data
        report_data = {
            "Session ID": session_id,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Hemorrhage Risk (%)": f"{pct:.1f}",
            "Risk Category": risk_cat,
            "Organ Failure": OF_num,
            "Pancreatic Fistula": pancreatic_fis,
            "Pus MDRO Infection": pan_MDRO,
            "Bloodstream Infection": blood_inf,
            "Age": age,
            "OF Duration (days)": OF_time,
            "Onset-to-intervention (days)": time_sur
        }

        # Download PDF button
        pdf_file = generate_pdf(report_data)
        st.download_button(
            label="🧾 Download PDF Report",
            data=pdf_file,
            file_name=f"IPN_hemorrhage_report_{session_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ============================================
# Right side (Now without Clinical Context)
# ============================================

with col_right:
    st.subheader("Model Overview")

    st.markdown(
        """
        **Outcome**  
        Probability of **intra-abdominal hemorrhage** in **infected pancreatic necrosis (IPN)**.

        **Predictor set**  
        - Organ failure (0/1/2)  
        - Postoperative pancreatic fistula  
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
