import os
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

# Small CSS tweaks for a cleaner, “app-like” look
st.markdown(
    """
    <style>
        /* Make the main page a bit wider and cleaner */
        .main {
            padding: 2rem 3rem;
        }
        /* Nice cards for results */
        .risk-card {
            border-radius: 0.75rem;
            padding: 1rem 1.5rem;
            margin-bottom: 0.75rem;
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
            margin-left: 0.25rem;
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
# Page title and intro
# =========================
st.title("🩸 Stacking Model for Bleeding Risk Prediction")
st.markdown(
    """
    This web application uses a stacking machine learning model to estimate the risk of 
    postoperative bleeding based on selected clinical features.  
    Enter the patient characteristics in the left sidebar and click **Predict** to obtain an 
    individualized risk estimate and model explanations based on SHAP.
    """
)

st.markdown(
    "<p class='small-muted'>This tool is intended for research and educational purposes only and "
    "should not replace clinical judgment.</p>",
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

    st.markdown("---")
    predict_btn = st.button("▶ Predict bleeding risk", use_container_width=True)

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


# =========================
# Main: prediction result
# =========================
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("Prediction Result")

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

            # Determine risk category
            if pct < 10:
                risk_cat = "Low"
                css_class = "risk-low"
                pill_class = "pill-low"
                risk_msg = "Low estimated risk of postoperative bleeding."
            elif pct < 50:
                risk_cat = "Intermediate"
                css_class = "risk-medium"
                pill_class = "pill-medium"
                risk_msg = "Intermediate risk. Close monitoring is recommended."
            else:
                risk_cat = "High"
                css_class = "risk-high"
                pill_class = "pill-high"
                risk_msg = "High risk. Consider intensified surveillance and early intervention."

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

            # Quick summary of the input profile
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

        except FileNotFoundError:
            st.error(
                "Model file `best_model_stack.pkl` was not found. "
                "Please upload the trained model to the app directory."
            )
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: **{e}**")
    else:
        st.info(
            "Set the patient features in the sidebar and click **Predict bleeding risk** "
            "to view the model output."
        )

with col_right:
    st.subheader("Model Overview")
    st.markdown(
        """
        - **Model type:** Stacking ensemble for binary classification  
        - **Outcome:** Postoperative bleeding (yes / no)  
        - **Inputs:** Organ failure status, pancreatic fistula, MDRO infection, 
          bloodstream infection, age, duration of organ failure, and onset-to-surgery interval.  

        The stacking model combines multiple base learners to improve robustness and 
        predictive performance compared with single algorithms.
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
    bleeding risk both at the level of base learners and for the final stacking model.
    """
)

tab1, tab2 = st.tabs(["Base learners", "Stacking model"])

with tab1:
    img1 = load_image("summary_plot.png")
    if img1 is not None:
        st.image(img1, caption="SHAP feature importance of base learners", use_column_width=True)
    else:
        st.warning("Image `summary_plot.png` not found in the app directory.")

with tab2:
    img2 = load_image("overall_shap.png")
    if img2 is not None:
        st.image(img2, caption="Global SHAP summary for the stacking model", use_column_width=True)
    else:
        st.warning("Image `overall_shap.png` not found in the app directory.")

st.markdown("---")
st.caption("© 2025 Bleeding Risk Stacking Model · Research prototype built with Streamlit")
