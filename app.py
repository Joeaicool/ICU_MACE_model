import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# =========================
# 1. 页面配置与高级医学 CSS
# =========================
st.set_page_config(page_title="Sepsis MACE Predictor", page_icon="🫀", layout="wide")

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .title-box { background: linear-gradient(135deg, #0A2540 0%, #B22222 100%); padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .title-box h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .title-box p { margin-top: 10px; font-size: 1.1rem; opacity: 0.9; }
    .card { background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 1.5rem; }
    .card-title { color: #0A2540; font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; border-bottom: 2px solid #F1F5F9; padding-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>Machine Learning for Predicting MACE Risk in Sepsis</h1>
    <p>An Interpretable Clinical Decision-Support Tool based on Routine Biomarkers</p>
</div>
""", unsafe_allow_html=True)

# 加载顶部医学横幅图
try:
    from PIL import Image
    if os.path.exists("1.png"):
        st.image(Image.open("1.png"), use_container_width=True)
except Exception:
    pass

# =========================
# 2. 全局配置与特征定义
# =========================
ML_MODEL_PATH = "RF_BO_best.pkl"
ALL_FEATURES = ["Temperature", "AST", "ALP", "Uric Acid", "Cr"]

# ⚠️ 关键步骤：硬编码训练集的 Mean 和 SD 进行手动标准化
# 依据你的 Baseline_Characteristics 表格总体数据
SCALING_PARAMS = {
    "Temperature": {"mean": 37.21, "sd": 1.10},
    "AST": {"mean": 99.00, "sd": 344.58},
    "ALP": {"mean": 131.48, "sd": 102.90},
    "Uric Acid": {"mean": 409.68, "sd": 203.12},
    "Cr": {"mean": 183.95, "sd": 171.03}
}

@st.cache_resource
def load_ml_model():
    try:
        return joblib.load(ML_MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Model loading failed. Please ensure '{ML_MODEL_PATH}' is in the same directory. Error: {e}")
        st.stop()

# =========================
# 3. UI: 临床数据输入区
# =========================
st.markdown('<div class="card"><div class="card-title">🩺 Enter Patient Baseline Biomarkers</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    val_temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.2, step=0.1)
    val_ast = st.number_input("AST (U/L)", min_value=0.0, max_value=2000.0, value=99.0, step=1.0)
with c2:
    val_alp = st.number_input("ALP (U/L)", min_value=0.0, max_value=1000.0, value=131.5, step=1.0)
    val_ua = st.number_input("Uric Acid (μmol/L)", min_value=0.0, max_value=1500.0, value=409.7, step=1.0)
with c3:
    val_cr = st.number_input("Creatinine (Cr, μmol/L)", min_value=0.0, max_value=2000.0, value=184.0, step=1.0)
    
st.markdown('</div>', unsafe_allow_html=True)

# 汇总原始输入数据
raw_inputs = {
    "Temperature": val_temp,
    "AST": val_ast,
    "ALP": val_alp,
    "Uric Acid": val_ua,
    "Cr": val_cr
}

# =========================
# 4. 推理与预测
# =========================
_, center_col, _ = st.columns([1, 2, 1])
if center_col.button("🚀 Calculate MACE Risk Probability", type="primary", use_container_width=True):
    
    with st.spinner("🤖 Processing data and running AI model..."):
        ml_model = load_ml_model()
        
        # 1. 对输入数据进行与训练集一致的 StandardScaler 变换
        scaled_inputs = {}
        for feat in ALL_FEATURES:
            m = SCALING_PARAMS[feat]["mean"]
            s = SCALING_PARAMS[feat]["sd"]
            scaled_inputs[feat] = (raw_inputs[feat] - m) / s
            
        # 2. 构建输入 DataFrame，顺序严格按照 ALL_FEATURES
        X_input = pd.DataFrame([[scaled_inputs[f] for f in ALL_FEATURES]], columns=ALL_FEATURES)
        
        # 3. 预测概率
        prob_pos = ml_model.predict_proba(X_input)[0][1] * 100

    # =========================
    # 5. 结果展示区
    # =========================
    st.markdown('<div class="card"><div class="card-title">📊 Step 2: Diagnostic Results</div>', unsafe_allow_html=True)
    res_c1, res_c2 = st.columns([1.2, 1])

    with res_c1:
        st.markdown('#### 🩺 Clinical Interpretation')
        # 根据实际临床需求，可将阈值调整为最佳截断值（此处暂时用常规的 50% 演示，你可以自己改为 20% 或 30% 等）
        if prob_pos >= 50:
            st.error("### ⚠️ High Risk of Subsequent MACE")
            st.write("The model indicates a **higher likelihood** that the patient will develop Major Adverse Cardiovascular Events.")
        else:
            st.success("### ✅ Low Risk of MACE")
            st.write("The model indicates a **lower likelihood** of short-term cardiovascular events.")
        st.info(f"**Calculated Probability of MACE:** **{prob_pos:.2f}%**")
        
        with st.expander("Show Scaled Inputs to Model"):
            st.write("Machine learning model requires standardized (Z-score) inputs for accurate prediction based on training distribution.")
            st.json({k: round(v, 4) for k, v in scaled_inputs.items()})

    with res_c2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob_pos, number={"suffix": "%", "font": {"size": 40}},
            title={"text": "MACE Risk Probability", "font": {"size": 18}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#B22222"},
                   "steps": [{"range": [0, 50], "color": "rgba(40, 167, 69, 0.2)"},
                             {"range": [50, 100], "color": "rgba(220, 53, 69, 0.2)"}]}
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # 6. SHAP 解释区
    # =========================
    st.markdown('<div class="card"><div class="card-title">🔍 AI Explainability (SHAP Waterfall)</div>', unsafe_allow_html=True)
    with st.spinner('Calculating SHAP values for personalized explanation...'):
        try:
            explainer = shap.TreeExplainer(ml_model)
            shap_values_raw = explainer.shap_values(X_input)
            
            if isinstance(shap_values_raw, list): sv_values = shap_values_raw[1][0]
            elif len(shap_values_raw.shape) == 3: sv_values = shap_values_raw[0, :, 1]
            else: sv_values = shap_values_raw[0]

            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)): base_val = base_val[1] if len(base_val)>1 else base_val[0]

            sv_in_plot = shap.Explanation(
                values=sv_values, base_values=float(base_val),
                data=X_input.iloc[0].values, # 这里传入标准化后的值，这是底层树模型的真实判断依据
                feature_names=ALL_FEATURES
            )

            p1, p2 = st.columns([1.5, 1])
            with p1:
                fig_wf, ax_wf = plt.subplots(figsize=(8, 5), dpi=150)
                shap.plots.waterfall(sv_in_plot, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig_wf)
                plt.close(fig_wf)

            with p2:
                st.markdown('#### Feature Impact')
                contrib_df = pd.DataFrame({
                    "Feature": ALL_FEATURES,
                    "Effect": ["⬆️ Drives towards MACE" if v > 0 else "⬇️ Protects from MACE" for v in sv_values],
                    "Impact (|SHAP|)": np.abs(sv_values)
                }).sort_values("Impact (|SHAP|)", ascending=False)
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"⚠️ SHAP explanation could not be rendered: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
