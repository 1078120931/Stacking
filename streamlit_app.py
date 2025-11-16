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
    page_title="Bleeding Risk in Infected Pancreatic Necrosis",
    layout="wide",
    page_icon="🩸",
)

# -------------------------
# Session-level patient / session ID
# -------------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()

session_id = st.session_state["session_id"]

# =========================
# Global style (CSS)
# =========================
st.markdown(
    """
    <style>
        .main {
            padding: 0rem 3rem 3rem 3rem;
        }
        /* Top header bar */
        .top-bar {
            background: linear-gradient(90deg, #0b7285, #1971c2);
            padding: 0.9rem 1.6rem;
            border-radius: 0 0 1.1rem 1.1rem;
            margin: -1.2rem -3rem 1.8rem -3rem;
            color: #ffffff;
        }
        .top-bar-title {
            font-size: 1.35rem;
            font-weight: 650;
        }
        .top-bar-subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
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
        section[data-testid="stSidebar"] {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Top bar (header like EASY-APP)
# =========================
header_html = f"""
<div class="top-bar">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div class="top-bar-title">Bleeding Risk Support in Infected Pancreatic Necrosis</div>
      <div class="top-bar-subtitle">
        Stacking ensemble · Research prototype · Gastrointestinal / intraluminal bleeding
      </div>
    </div>
    <div class="top-bar-right">
      <div>Logged in as: <b>Guest</b></div>
      <div>Session ID: {session_id}</div>
      <div>{datetime.now().strftime("%Y-%m-%d")}</div>
    </div>
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# =========================
# Page intro
# =========================
st.title("🩸 Stacking Model for Bleeding Risk in Infected Pancreatic Necrosis")

st.markdown(
    """
    This web application uses a stacking machine learning model to estimate the risk of 
    **clinically relevant bleeding** in patients with **infected pancreatic necrosis (IPN)**, 
    based on routinely collected clinical features.

    Enter the patient characteristics in the left sidebar and click **Predict bleeding risk** 
    to obtain an individualized risk estimate and visual model explanations based on SHAP.
    """
)

st.markdown(
    "<p class='small-muted'>This tool is intended for research and educational purposes only and "
    "must not be used as a stand-alone basis for clinical decisions.</p>",
    unsafe_allow_html=True,
)

# =========================
# Sidebar: input features
# =========================
with st.sidebar:
    st.header("Input Features (IPN Cohort)")

    OF_num = st.selectbox(
        "Organ failure (0=None, 1=Single, 2=Multiple)",
        options=[0, 1, 2],
        index=0,
        help="Maximum number of organ failures during the IPN course.",
    )

    pancreatic_fis = st.selectbox(
        "Postoperative pancreatic fistula (0=No, 1=Yes)",
        options=[0, 1],
        index=0,
    )

    pan_MDRO = st.selectbox(
        "Pus MDRO infection (0=No, 1=Yes)",
        options=[0, 1],
        index=0,
        help="Presence of multidrug-resistant organisms in pancreatic / peripancreatic pus cultures.",
    )

    blood_inf = st.selectbox(
        "Bloodstream infection (0=No, 1=Yes)",
        options=[0, 1],
        index=0,
    )

    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=120,
        value=60,
        step=1,
    )

    OF_time = st.number_input(
        "Duration of organ failure (days)",
        min_value=0,
        max_value=365,
        value=0,
        step=1,
    )

    time_sur = st.number_input(
        "Onset-to-intervention interval (days)",
        min_value=0,
        max_value=365,
        value=0,
        step=1,
        help="Time from onset of acute pancreatitis to the first invasive intervention for IPN.",
    )

    st.markdown("---")
    predict_btn = st.button("▶ Predict bleeding risk", use_container_width=True)
    reset_btn = st.button("⟲ Reset session", use_container_width=True)

# Reset: regenerate session ID
if reset_btn:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()
    st.experimental_rerun()

# =========================
# Utilities
# =========================
@st.cache_resource(show_spinner="Loading stacking model...")
def load_model(path: str = "best_model_stack.pkl"):
    model = joblib.load(path)
    return model


def load_image(path: str):
    if os.path.exists(path):
        return Image.open(path)
    return None


def build_export_csv(record: dict) -> str:
    """Build a one-line CSV string with header for download."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(record.keys()))
    writer.writeheader()
    writer.writerow(record)
    return buffer.getvalue()


# =========================
# Layout for main content
# =========================
col_left, col_right = st.columns([1.1, 1])

# ---------- Left: prediction ----------
with col_left:
    st.subheader("Prediction Result")

    prediction_made = False
    export_data = None

    if predict_btn:
        try:
            model = load_model()

            X = np.array(
                [[OF_num, pancreatic_fis, pan_MDRO, blood_inf, age, OF_time, time_sur]]
            )

            # Binary classification: probability of bleeding = class 1
            prob = float(model.predict_proba(X)[0][1])
            prob = max(0.0, min(prob, 1.0))  # safety clip
            pct = prob * 100

            # Fixed thresholds
            low_thr = LOW_RISK_THRESHOLD
            high_thr = HIGH_RISK_THRESHOLD

            if pct < low_thr:
                risk_cat = "Low"
                css_class = "risk-low"
                pill_class = "pill-low"
                risk_msg = (
                    "Low estimated risk of clinically relevant bleeding during the IPN course."
                )
            elif pct < high_thr:
                risk_cat = "Intermediate"
                css_class = "risk-medium"
                pill_class = "pill-medium"
                risk_msg = (
                    "Intermediate bleeding risk. Close monitoring and optimisation of correctable "
                    "factors are recommended."
                )
            else:
                risk_cat = "High"
                css_class = "risk-high"
                pill_class = "pill-high"
                risk_msg = (
                    "High bleeding risk. Consider intensified surveillance and timely diagnostic "
                    "or therapeutic interventions according to local practice."
                )

            # Result card
            st.markdown(
                f"""
                <div class="risk-card {css_class}">
                    <h4 style="margin-top:0;">Predicted bleeding risk: {pct:.1f}% 
                        <span class="pill-label {pill_class}">{risk_cat} risk</span>
                    </h4>
                    <p style="margin-bottom:0;">{risk_msg}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Progress bar
            st.progress(prob)

            # Quick summary
            with st.expander("Show input feature summary"):
                st.write(
                    {
                        "Organ failure (0/1/2)": OF_num,
                        "Postoperative pancreatic fistula (0/1)": pancreatic_fis,
                        "Pus MDRO infection (0/1)": pan_MDRO,
                        "Bloodstream infection (0/1)": blood_inf,
                        "Age (years)": age,
                        "Duration of organ failure (days)": OF_time,
                        "Onset-to-intervention interval (days)": time_sur,
                    }
                )

            # Prepare exportable data
            export_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "bleeding_probability": f"{prob:.4f}",
                "bleeding_risk_percent": f"{pct:.1f}",
                "risk_category": risk_cat,
                "OF_num": OF_num,
                "postoperative_pancreatic_fistula": pancreatic_fis,
                "pus_MDRO_infection": pan_MDRO,
                "bloodstream_infection": blood_inf,
                "age": age,
                "OF_duration_days": OF_time,
                "onset_to_intervention_days": time_sur,
                "low_threshold_percent": low_thr,
                "high_threshold_percent": high_thr,
            }
            prediction_made = True

        except FileNotFoundError:
            st.error(
                "Model file `best_model_stack.pkl` was not found. "
                "Please upload the trained model to the app directory."
            )
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: **{e}**")
    else:
        st.info(
            "Set the patient features in the sidebar and click "
            "**Predict bleeding risk** to view the model output."
        )

    # Download button (CSV)
    if prediction_made and export_data is not None:
        csv_content = build_export_csv(export_data)
        st.download_button(
            "💾 Download prediction as CSV",
            data=csv_content,
            file_name=f"IPN_bleeding_risk_{session_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------- Right: model overview ----------
with col_right:
    st.subheader("Model & Interpretation (IPN Cohort)")
    st.markdown(
        f"""
        **Clinical context**  
        Patients with **infected pancreatic necrosis (IPN)** are at substantial risk of 
        gastrointestinal or intraluminal bleeding, especially in the presence of organ failure,
        infection with multidrug-resistant organisms, and complex postoperative courses.

        **Model type**  
        Stacking ensemble for binary classification of clinically relevant bleeding events.

        **Current predictor set**  
        - Organ failure status (none / single / multiple)  
        - Postoperative pancreatic fistula  
        - Pus MDRO infection  
        - Bloodstream infection  
        - Age  
        - Duration of organ failure  
        - Onset-to-intervention interval  

        **Risk stratification (fixed thresholds)**  
        - Low risk: predicted probability **< {LOW_RISK_THRESHOLD:.0f}%**  
        - Intermediate risk: **{LOW_RISK_THRESHOLD:.0f}–{HIGH_RISK_THRESHOLD:.0f}%**  
        - High risk: **≥ {HIGH_RISK_THRESHOLD:.0f}%**
        """
    )

    with st.expander("How should the predicted risk be interpreted?"):
        st.markdown(
            """
            - The **percentage value** corresponds to the model-estimated probability of 
              clinically relevant bleeding during the IPN course.  
            - **Low risk** patients may be candidates for standard monitoring if consistent 
              with the overall clinical picture.  
            - **Intermediate risk** suggests the need for closer observation and optimisation 
              of haemodynamic status, anticoagulation, and infection control.  
            - **High risk** should prompt careful evaluation for early imaging, endoscopic or 
              interventional radiologic procedures, or multidisciplinary discussion, 
              according to local guidelines and clinical judgement.
            """
        )

# =========================
# SHAP visualisation
# =========================
st.markdown("---")
st.header("🔍 SHAP-Based Model Explanation")

st.markdown(
    """
    SHAP (SHapley Additive exPlanations) values quantify the contribution of each feature to the
    model prediction. The plots below summarise how individual variables influence the estimated 
    bleeding risk, both at the level of base learners and for the final stacking model.
    """
)

tab1, tab2 = st.tabs(["Base learners", "Stacking model"])

with tab1:
    img1 = load_image("summary_plot.png")
    if img1 is not None:
        st.image(
            img1,
            caption="SHAP feature importance of base learners in the first layer of the stacking model",
            use_column_width=True,
        )
    else:
        st.warning("Image `summary_plot.png` not found in the app directory.")

with tab2:
    img2 = load_image("overall_shap.png")
    if img2 is not None:
        st.image(
            img2,
            caption="Global SHAP summary for the final stacking model",
            use_column_width=True,
        )
    else:
        st.warning("Image `overall_shap.png` not found in the app directory.")

st.markdown("---")
st.caption("© 2025 Infected Pancreatic Necrosis Bleeding Risk · Stacking model prototype built with Streamlit")
