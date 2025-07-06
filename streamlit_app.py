import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

# ---------- 页面基本设置 ----------
st.set_page_config(
    page_title="Stacking 出血风险预测",
    layout="wide",
    page_icon="🩸"
)
st.title("🩸 Stacking 模型出血风险预测与 SHAP 可视化")
st.write("请输入相关特征，预测个体的出血风险，并查看各因素对模型结果的贡献。")

# ---------- 侧边栏输入 ----------
with st.sidebar:
    st.header("输入特征")
    OF_num         = st.selectbox("Organ failure (0=None,1=Single,2=Multi)", [0, 1, 2])
    pancreatic_fis = st.selectbox("Pancreatic fistula (0=No,1=Yes)", [0, 1])
    pan_MDRO       = st.selectbox("Pus MDRO infection (0=No,1=Yes)", [0, 1])
    blood_inf      = st.selectbox("Blood infection (0=No,1=Yes)", [0, 1])
    age            = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    OF_time        = st.number_input("OF duration (days)", min_value=0, max_value=365, value=0)
    time_sur       = st.number_input("Onset-to-surgery (days)", min_value=0, max_value=365, value=0)
    predict_btn    = st.button("进行预测")

# ---------- 延迟加载 & 缓存模型 ----------
@st.cache_resource(show_spinner="📦 正在加载模型…")
def load_model(path="best_model_stack.pkl"):
    return joblib.load(path)

# ---------- 预测 ----------
if predict_btn:
    try:
        model = load_model()
        X = np.array([[OF_num, pancreatic_fis, pan_MDRO, blood_inf,
                       age, OF_time, time_sur]])

        # 如果是分类模型，使用 predict_proba
        prob = model.predict_proba(X)[0][1]
        pct = prob * 100
        st.success(f"🩸 预测出血风险：{pct:.1f}%")

        # 显示风险等级标签
        if pct < 10:
            st.info("🔵 低风险")
        elif pct < 30:
            st.warning("🟡 中风险，请密切监测")
        else:
            st.error("🔴 高风险，建议积极干预")

        # 显示进度条
        st.progress(min(prob, 1.0))

    except Exception as e:
        st.error(f"❌ 预测发生错误：{e}")

# ---------- SHAP 可视化 ----------
st.header("🔍 SHAP 可视化分析")
st.write("以下图表展示了模型的 SHAP 分析结果，解释各特征对出血风险预测的贡献。")

def show_img(path, caption):
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=True)
    else:
        st.warning(f"⚠️ 未找到图像文件：{path}")

show_img("summary_plot.png", "第一层基学习器 SHAP 贡献分析")
show_img("overall_shap.png", "整体 Stacking 模型 SHAP 贡献分析")

st.markdown("---")
st.caption("© 2025 出血风险预测模型演示 · Powered by Streamlit")
