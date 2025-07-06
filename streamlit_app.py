import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

# ---------- 页面基本设置 ----------
st.set_page_config(
    page_title="Stacking 模型预测与 SHAP 可视化",
    layout="wide",
    page_icon="📊"
)
st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("通过输入特征值进行模型预测，并结合 SHAP 了解各特征的贡献。")

# ---------- 侧边栏输入 ----------
with st.sidebar:
    st.header("请输入特征值")
    OF_num        = st.selectbox("Organ failure (0=None,1=Single,2=Multi)", [0, 1, 2])
    pancreatic_fis= st.selectbox("Pancreatic fistula (0=No,1=Yes)", [0, 1])
    pan_MDRO      = st.selectbox("Pus MDRO infection (0=No,1=Yes)", [0, 1])
    blood_inf     = st.selectbox("Blood infection (0=No,1=Yes)", [0, 1])
    age           = st.number_input("Age (years)", 0, 120, 30)
    OF_time       = st.number_input("OF duration (days)", 0, 365, 0)
    time_sur      = st.number_input("Onset-to-surgery (days)", 0, 365, 0)
    predict_btn   = st.button("进行预测")

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
        y_pred = model.predict(X)[0]
        st.success(f"✅ 预测结果：{y_pred:.2f}")
    except Exception as e:
        st.error(f"❌ 预测发生错误：{e}")

# ---------- SHAP 可视化 ----------
st.header("SHAP 可视化分析")

def show_img(path, caption):
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=True)
    else:
        st.warning(f"⚠️ 未找到图像文件：{path}")

show_img("summary_plot.png", "第一层基学习器 SHAP 贡献分析")
show_img("overall_shap.png",  "整体 Stacking 模型 SHAP 贡献分析")  # ← 改成新文件名

st.markdown("---")
st.caption("© 2025 Stacking Model Demo")
