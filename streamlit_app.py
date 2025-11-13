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
# Page configuration
# =========================
st.set_page_config(
    page_title="Stacking Bleeding Risk Calculator",
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
        /* Make sidebar inputs a bit nicer */
        section[data-testid="stSidebar"] {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Top bar (like EASY-APP header)
# =========================
header_html = f"""
<div class="top-bar">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div class="top-bar-title">Bleeding Risk Decision Support</div>
      <div class="top-bar-subtitle">
        Stacking ensemble · Research prototype · Postoperative bleeding
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
st.title("🩸 Stacking Model for Bleeding Risk Prediction")

st.markdown(
    """
    This web application uses a stacking machine learning model to estimate the risk of 
    postoperative bleeding based on selected clinical features.

    Enter the patient characteristics in the left sidebar and click **Predict bleeding risk** 
    to obtain an individualized risk estimate and visual explanations based on SHAP.
    """
)

st.markdown(
    "<p class='small-muted'>This tool is intended for research and educational purposes only and "
    "should not replace clinical judgement.</p>",
    unsafe_allow_html=True,
)

# =========================
# Sidebar: input features
# =========================
with st.sidebar:
    st.header("Input Features")

    OF_num = st.selectbox(
        "Organ failure (0=None, 1=Single, 2=Multiple)",
        options=[0, 1, 2],
        index=0,
        help="Highest number of organ failures during the perioperative period.",
    )

    pancreatic_fis = st.selectbox(
        "Pancreatic fistula (0=No, 1=Yes)",
        options=[0, 1],
        index=0,
    )

    pan_MDRO = st.selectbox(
        "Pus MDRO infection (0=No, 1=Yes)",
        options=[0, 1],
        index=0,
        help="Presence of multidrug-resistant organism (MDRO) in pus cultures.",
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
        "Onset-to-surgery interval (days)",
        min_value=0,
        max_value=365,
        value=0,
        step=1,
        help="Time from disease onset to definitive surgery.",
    )

    # ---- Advanced settings: adjustable thresholds ----
    with st.expander("Risk thresholds (advanced)"):
        low_thr = st.slider(
            "Upper limit for low risk (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
        )
        high_thr = st.slider(
            "Upper limit for intermediate risk (%)",
            min_value=low_thr,
            max_value=100.0,
            value=50.0,
            step=1.0,
        )
        st.caption(
            "Low risk: < low threshold · "
            "Intermediate risk: between low and high thresholds · "
            "High risk: ≥ high threshold."
        )

    st.markdown("---")
    predict_btn = st.button("▶ Predict bleeding risk", use_container_width=True)
    reset_btn = st.button("⟲ Reset session", use_container_width=True)

# Reset: simply regenerate session ID and rerun
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

            # Determine risk category based on adjustable thresholds
            if pct < low_thr:
                risk_cat = "Low"
                css_class = "risk-low"
                pill_class = "pill-low"
                risk_msg = "Low estimated risk of postoperative bleeding."
            elif pct < high_thr:
                risk_cat = "Intermediate"
                css_class = "risk-medium"
                pill_class = "pill-medium"
                risk_msg = "Intermediate risk. Close monitoring is recommended."
            else:
                risk_cat = "High"
                css_class = "risk-high"
                pill_class = "pill-high"
                risk_msg = (
                    "High risk. Consider intensified surveillance and timely intervention."
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
                        "Pancreatic fistula (0/1)": pancreatic_fis,
                        "Pus MDRO infection (0/1)": pan_MDRO,
                        "Bloodstream infection (0/1)": blood_inf,
                        "Age (years)": age,
                        "Duration of organ failure (days)": OF_time,
                        "Onset-to-surgery interval (days)": time_sur,
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
                "pancreatic_fistula": pancreatic_fis,
                "pus_MDRO_infection": pan_MDRO,
                "bloodstream_infection": blood_inf,
                "age": age,
                "OF_duration_days": OF_time,
                "onset_to_surgery_days": time_sur,
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
            file_name=f"bleeding_risk_{session_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------- Right: model overview & thresholds ----------
with col_right:
    st.subheader("Model & Threshold Overview")
    st.markdown(
        """
        **Model type**  
        Stacking ensemble for binary classification of postoperative bleeding.

        **Outcome**  
        Probability of clinically relevant postoperative bleeding (yes / no).

        **Predictor set (current version)**  
        - Organ failure status (none / single / multiple)  
        - Presence of pancreatic fistula  
        - Pus MDRO infection  
        - Bloodstream infection  
        - Age  
        - Duration of organ failure  
        - Onset-to-surgery interval  

        The model combines multiple base learners and is calibrated on an internal cohort.  
        Thresholds for risk categories can be adjusted in the sidebar to match local practice 
        or specific study protocols.
        """
    )

    with st.expander("How to interpret the risk estimate?"):
        st.markdown(
            """
            - The **percentage value** represents the model-estimated probability of bleeding.  
            - **Low risk** usually corresponds to patients suitable for routine monitoring.  
            - **Intermediate risk** suggests closer observation or optimisation of modifiable factors.  
            - **High risk** may justify intensified surveillance, early imaging, or proactive intervention, 
              depending on local guidelines and clinical judgement.
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
    bleeding risk both at the level of base learners and for the aggregated stacking model.
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
st.caption("© 2025 Bleeding Risk Stacking Model · Research prototype built with Streamlit")
