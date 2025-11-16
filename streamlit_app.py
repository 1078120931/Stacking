# ============================================
# Imports（每行都标注用途）
# ============================================

import os                  # 文件与路径操作（检查图片、模型文件是否存在）
import uuid                # 生成唯一的 session ID，方便区分每次预测
import csv                 # 预留：如需导出 CSV 报告可使用（当前未强制使用）
from io import StringIO    # 预留：如需在内存中构建 CSV 文本可使用
from datetime import datetime  # 获取当前日期与时间，用于报告与页面显示

import numpy as np         # 处理数值数组，构建模型输入 X
import joblib              # 加载训练好的 stacking 模型（.pkl 文件）
from PIL import Image      # 加载 PNG 格式的 SHAP 图像
import streamlit as st     # 构建 Web 界面的核心库


# ============================================
# 页面基础配置
# ============================================

st.set_page_config(
    page_title="IPN Hemorrhage Risk — Xiangya Hospital",
    layout="wide",
    page_icon="🩸",
)

# -------------------------
# Session-level patient / session ID
# -------------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()
session_id = st.session_state["session_id"]


# ============================================
# 全局样式（CSS）
# ============================================

st.markdown(
    """
    <style>
        .main {
            padding: 0rem 3rem 3rem 3rem;
        }
        /* 顶部色条 */
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
        /* 风险结果卡片 */
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
           背景: #fff8e1;
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
        .pill-low  { background:#e8f5e9; color:#2e7d32; }
        .pill-medium { background:#fff8e1; color:#f9a825; }
        .pill-high { background:#ffebee; color:#c62828; }
        .small-muted {
            color: #777777;
            font-size: 0.8rem;
        }
        /* 侧边栏上方空一点 */
        section[data-testid="stSidebar"] {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# 顶部栏（类似 EASY-APP header）
# ============================================

header_html = f"""
<div class="top-bar">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div class="top-bar-title">IPN Hemorrhage Decision Support</div>
      <div class="top-bar-subtitle">
        Stacking ensemble · Research prototype · Intra-abdominal hemorrhage in IPN
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


# ============================================
# Page intro
# ============================================

st.title("🩸 Stacking Model for Hemorrhage Risk Prediction in Infected Pancreatic Necrosis")

st.markdown(
    """
    This web application uses a stacking machine learning model to estimate the risk of 
    **clinically significant intra-abdominal hemorrhage** in patients with 
    **infected pancreatic necrosis (IPN)**.

    Enter the patient characteristics in the left sidebar and click 
    **Predict hemorrhage risk** to obtain an individualized risk estimate and visual explanations based on SHAP.
    """
)

st.markdown(
    "<p class='small-muted'>This tool is intended for research and educational purposes only and "
    "should not replace clinical judgement.</p>",
    unsafe_allow_html=True,
)


# ============================================
# 侧边栏输入
# ============================================

with st.sidebar:
    st.header("Input Features (IPN)")

    OF_num = st.selectbox(
        "Organ failure (0=None, 1=Single, 2=Multiple)",
        options=[0, 1, 2],
        index=0,
        help="Highest number of organ failures during the IPN disease course.",
    )

    # 改成 Pancreatic fistula，不再写 postoperative
    pancreatic_fis = st.selectbox(
        "Pancreatic fistula (0=No, 1=Yes)",
        options=[0, 1],
        index=0,
    )

    pan_MDRO = st.selectbox(
        "Pus MDRO infection (0=No, 1=Yes)",
        options=[0, 1],
        index=0,
        help="MDRO identified in pancreatic or peripancreatic infected collections.",
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
        help="Time from AP onset to the first invasive intervention for IPN.",
    )

    st.markdown("---")
    predict_btn = st.button("▶ Predict hemorrhage risk", use_container_width=True)
    reset_btn = st.button("⟲ Reset session", use_container_width=True)

# Reset: simply regenerate session ID and rerun
if reset_btn:
    st.session_state["session_id"] = "S-" + uuid.uuid4().hex[:8].upper()
    st.experimental_rerun()


# ============================================
# Utilities：加载模型、图片
# ============================================

@st.cache_resource(show_spinner="Loading stacking model...")
def load_model(path: str = "best_model_stack.pkl"):
    model = joblib.load(path)
    return model


def load_image(path: str):
    if os.path.exists(path):
        return Image.open(path)
    return None


# ============================================
# 纯 Python 生成较美观的单页 PDF（带行距，不重叠）
# ============================================

def _pdf_escape(text: str) -> str:
    """转义 PDF 文本中的特殊字符"""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_pdf(data: dict) -> bytes:
    """
    生成一个简洁单页 PDF：
    - 顶部标题两行
    - 下面按行距 14pt 逐行打印 key: value
    不依赖第三方库，适合 Streamlit Cloud 环境。
    """
    lines = [
        "Xiangya Hospital",
        "IPN Intra-Abdominal Hemorrhage Risk Report",
        "",  # 空行
    ]
    for k, v in data.items():
        lines.append(f"{k}: {v}")

    content_lines = []
    content_lines.append("BT")
    content_lines.append("/F1 12 Tf")
    content_lines.append("14 TL")               # 设置行距 14pt
    content_lines.append("1 0 0 1 50 800 Tm")   # 文本起始位置 (x=50, y=800)

    first = True
    for line in lines:
        if first:
            content_lines.append(f"({_pdf_escape(line)}) Tj")
            first = False
        else:
            content_lines.append("T*")  # 按 TL 往下移一行
            content_lines.append(f"({_pdf_escape(line)}) Tj")

    content_lines.append("ET")
    stream_content = "\n".join(content_lines).encode("latin-1")

    # 对象定义
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    obj4 = (
        b"4 0 obj\n<< /Length "
        + str(len(stream_content)).encode("ascii")
        + b" >>\nstream\n"
        + stream_content
        + b"\nendstream\nendobj\n"
    )
    obj5 = b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    objects = [obj1, obj2, obj3, obj4, obj5]

    header = b"%PDF-1.4\n"
    offsets = []
    current_offset = len(header)

    for obj in objects:
        offsets.append(current_offset)
        current_offset += len(obj)

    xref_offset = current_offset
    xref_entries = [b"xref\n0 6\n", b"0000000000 65535 f \n"]
    for off in offsets:
        xref_entries.append(f"{off:010d} 00000 n \n".encode("ascii"))
    xref = b"".join(xref_entries)

    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )

    pdf_bytes = header + b"".join(objects) + xref + trailer
    return pdf_bytes


# ============================================
# Layout for main content
# ============================================

col_left, col_right = st.columns([1.1, 1])

# ---------- Left: prediction ----------
with col_left:
    st.subheader("Prediction Result")

    if predict_btn:
        try:
            model = load_model()

            X = np.array(
                [[OF_num, pancreatic_fis, pan_MDRO, blood_inf, age, OF_time, time_sur]]
            )

            # Binary classification: probability of hemorrhage = class 1
            prob = float(model.predict_proba(X)[0][1])
            prob = max(0.0, min(prob, 1.0))  # safety clip
            pct = prob * 100

            # 固定阈值：<10 低，10–50 中，≥50 高
            if pct < 10:
                risk_cat = "Low"
                css_class = "risk-low"
                pill_class = "pill-low"
                risk_msg = (
                    "Low estimated risk of clinically significant intra-abdominal hemorrhage in IPN."
                )
            elif pct < 50:
                risk_cat = "Intermediate"
                css_class = "risk-medium"
                pill_class = "pill-medium"
                risk_msg = (
                    "Intermediate risk of intra-abdominal hemorrhage. Close monitoring is recommended."
                )
            else:
                risk_cat = "High"
                css_class = "risk-high"
                pill_class = "pill-high"
                risk_msg = (
                    "High risk of clinically significant intra-abdominal hemorrhage. "
                    "Consider early vascular evaluation, imaging, and timely intervention."
                )

            # Result card
            st.markdown(
                f"""
                <div class="risk-card {css_class}">
                    <h4 style="margin-top:0;">Predicted hemorrhage risk: {pct:.1f}% 
                        <span class="pill-label {pill_class}">{risk_cat} risk</span>
                    </h4>
                    <p style="margin-bottom:0;">{risk_msg}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 显示进度条
            st.progress(prob)

            # 生成 PDF 报告的数据（这里的 Key 也统一成 Pancreatic fistula）
            report_data = {
                "Session ID": session_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Hemorrhage risk (%)": f"{pct:.1f}",
                "Risk category": risk_cat,
                "Organ failure (0/1/2)": OF_num,
                "Pancreatic fistula (0/1)": pancreatic_fis,
                "Pus MDRO infection (0/1)": pan_MDRO,
                "Bloodstream infection (0/1)": blood_inf,
                "Age (years)": age,
                "OF duration (days)": OF_time,
                "Onset-to-intervention (days)": time_sur,
            }

            pdf_bytes = generate_pdf(report_data)

            st.download_button(
                "🧾 Download PDF report",
                data=pdf_bytes,
                file_name=f"IPN_hemorrhage_report_{session_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
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
            "Set the patient features in the sidebar and click "
            "**Predict hemorrhage risk** to view the model output."
        )


# ---------- Right: model overview ----------
with col_right:
    st.subheader("Model Overview (IPN Hemorrhage)")
    st.markdown(
        """
        **Outcome**  
        Probability of **intra-abdominal hemorrhage** in patients with **infected pancreatic necrosis (IPN)**.

        **Predictor set (current version)**  
        - Organ failure status (none / single / multiple)  
        - Pancreatic fistula  
        - Pus MDRO infection  
        - Bloodstream infection  
        - Age  
        - Duration of organ failure  
        - Onset-to-intervention interval  
        """
    )


# ============================================
# SHAP visualisation
# ============================================

st.markdown("---")
st.header("🔍 SHAP-Based Model Explanation")

st.markdown(
    """
    SHAP (SHapley Additive exPlanations) values quantify the contribution of each feature to the
    predicted risk of intra-abdominal hemorrhage in IPN.
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
            caption="Global SHAP summary for the final stacking model (IPN hemorrhage)",
            use_column_width=True,
        )
    else:
        st.warning("Image `overall_shap.png` not found in the app directory.")

st.markdown("---")
st.caption("© 2025 Xiangya Hospital · IPN Intra-Abdominal Hemorrhage Prediction System")
