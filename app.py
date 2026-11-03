"""MACE research prediction app. Start with `streamlit run app.py`."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from predictor_core import (
    MODELS,
    DeploymentError,
    calculate_shap,
    load_model,
    load_preprocessing,
    make_features,
    predict_score,
    trained_features,
)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AI Clinical Risk Predictor | Precision Medicine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background-color: #F8F9FA;}
    .title-box {background: linear-gradient(135deg, #0A2540, #1750A1);
        padding: 1.9rem; border-radius: 12px; color: white;
        margin-bottom: 1.5rem; text-align: center;}
    .title-box h1 {color: white; margin: 0; font-size: 2.15rem;}
    .title-box p {color: #E6F0FF; font-size: 1.05rem; margin: .7rem 0 0;}
    .clinical-note {background: #EBF4FA; color: #0A2540;
        padding: 1.1rem 1.5rem; border-radius: 8px;
        border-left: 5px solid #1750A1; margin-bottom: 1.5rem;}
    .footer {margin-top: 3rem; padding-top: 1rem;
        border-top: 1px solid #E2E8F0; color: #64748B;
        font-size: .9rem; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Model loading is cached for speed but MUST remain read-only.
@st.cache_resource(show_spinner="Loading model...")
def cached_load_model(path: str):
    return load_model(path)


st.sidebar.markdown("## ⚙️ Model Selection")
st.sidebar.caption("Select a fitted AI architecture for the same clinical endpoint.")
selected = st.sidebar.radio(
    "Model",
    list(MODELS),
    index=0,
    label_visibility="collapsed",
)
model_filename = MODELS[selected]
model_path = BASE_DIR / model_filename
st.sidebar.info(f"**Currently Active:**\n\n{selected}\n\n`{model_filename}`")
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype. Not validated for independent clinical use.")

logo_path = BASE_DIR / "images.jpeg"
if logo_path.is_file():
    st.sidebar.image(str(logo_path), use_container_width=True)

st.markdown(
    """
    <div class="title-box">
      <h1>Advanced Machine Learning Model for Clinical Risk Prediction</h1>
      <p>Explainable Artificial Intelligence (XAI) for Personalized Patient Assessment</p>
    </div>
    <div class="clinical-note">
      <b>Objective:</b> This tool evaluates a fitted ICU MACE prediction model from
      five clinical biomarkers using preprocessing learned from its training cohort.
      <br><b>Important:</b> Model output is a research prediction score and should not
      be interpreted as a calibrated event probability or replace clinical assessment.
    </div>
    """,
    unsafe_allow_html=True,
)

if logo_path.is_file():
    st.image(str(logo_path), use_container_width=True)

try:
    model = cached_load_model(str(model_path))
    features = trained_features(model)
    # Fail closed unless this config was exported from the ORIGINAL training scaler.
    cfg = load_preprocessing(BASE_DIR / "preprocessing.json", model, model_path)
except DeploymentError as exc:
    st.error(f"Deployment configuration incomplete: {exc}")
    with st.expander("How to repair deployment"):
        st.markdown(
            "Copy the original fitted StandardScaler's `mean_` and `scale_` "
            "for the Boruta-selected features into `preprocessing.json`, using "
            "`export_preprocessing.py` in the same training run. No arbitrary "
            "reference ranges, new scaler fits, or guessed parameters are safe. "
            "Read `README.md` for the exact snippet."
        )
    st.stop()

st.subheader("📝 Step 1: Patient Biomarker Input")
st.caption(
    "Enter the **original laboratory/clinical values**, in exactly the same "
    "units as the model's training dataset. All five values are required. "
    "No values are assumed or imputed in this web interface."
)

user_input = {}
with st.form("patient_form", clear_on_submit=False, border=True):
    columns = st.columns(3)
    for idx, f in enumerate(features):
        unit = cfg["units"].get(f)
        label = f"{f} ({unit})" if unit else f"{f} (training unit)"
        with columns[idx % 3]:
            user_input[f] = st.number_input(
                label,
                min_value=None,
                max_value=None,
                value=None,
                step=0.1,
                format="%.4f",
                key=f"biomarker_{f}",
                help="Use the original unscaled value and the original training unit.",
            )
    submitted = st.form_submit_button(
        f"🚀 Run Assessment with {selected.split(' ')[0]}",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.subheader("📊 Step 2: Prediction Results")
    try:
        if any(v is None for v in user_input.values()):
            raise DeploymentError("Please enter all five biomarkers before prediction.")
        raw, scaled = make_features(user_input, cfg)
        score = predict_score(model, scaled)
    except DeploymentError as exc:
        st.error(str(exc))
        st.stop()

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.markdown("#### 🩺 Model Assessment")
        st.metric("Model prediction score (MACE=1)", f"{score * 100:.4f}%")
        threshold = cfg["validated_threshold"]
        if threshold is None:
            st.info(
                "No validated clinical cutoff was supplied. The app intentionally "
                "does not declare High Risk or Low Risk based on an arbitrary 50% threshold."
            )
        else:
            status = "Above" if score >= threshold else "Below"
            st.write(
                f"**{status} prespecified decision threshold:** "
                f"{threshold * 100:.2f}%"
            )
            st.caption(f"Threshold validation reference: {cfg['threshold_source']}")
        st.caption(
            "The model score may be uncalibrated. It is not a confirmed individual's "
            "real-world MACE event probability."
        )
    with c2:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={"suffix": "%", "valueformat": ".2f", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1750A1"},
                    "bgcolor": "#EDF2F7",
                    # Do not paint arbitrary green/amber/red clinical risk bands.
                },
            )
        )
        gauge.update_layout(height=265, margin=dict(l=20, r=20, t=25, b=20))
        st.plotly_chart(gauge, use_container_width=True)

    with st.expander("Input verification — original vs training-scale values"):
        check = pd.DataFrame({
            "Feature": features,
            "Patient raw value": raw.iloc[0].tolist(),
            "Model input (scaled)": scaled.iloc[0].tolist(),
            "Unit": [cfg["units"].get(f, "Not documented") for f in features],
        })
        st.dataframe(check, hide_index=True, use_container_width=True)
        st.caption("The scaler comes from the original training cohort, not this patient.")

    st.subheader("🔍 Step 3: AI Explainability (SHAP Analysis)")
    st.caption(
        "SHAP explains this exact prediction. Positive contributions increase the "
        "model's MACE=1 output. The mathematical output scale is noted below."
    )

    try:
        with st.spinner("Calculating SHAP contributions..."):
            explanation, scale_label, reference_label = calculate_shap(
                model, scaled, raw, score
            )
        st.caption(
            f"Explanation scale: **{scale_label}**. "
            f"Reference: {reference_label}. "
            "For XGBoost, log-odds contributions are not percentage-point changes."
        )
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.markdown("**SHAP Waterfall Plot**")
            plt.close("all")
            shap.plots.waterfall(explanation, max_display=len(features), show=False)
            fig = plt.gcf()
            fig.set_size_inches(8.2, 5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("**Patient-Specific Feature Contributions**")
            contributions = np.asarray(explanation.values, dtype=float)
            importance = np.abs(contributions)
            pct = 100 * importance / importance.sum() if importance.sum() else np.zeros_like(importance)
            ranking = pd.DataFrame({
                "Biomarker": features,
                "Patient Value": raw.iloc[0].to_numpy(dtype=float),
                "Direction": [
                    "↑ Increased model output" if v > 0 else (
                        "↓ Decreased model output" if v < 0 else "No effect"
                    ) for v in contributions
                ],
                "SHAP value": contributions,
                "Absolute share (%)": pct,
            }).sort_values("Absolute share (%)", ascending=False)
            st.dataframe(
                ranking.style.format({
                    "Patient Value": "{:.2f}",
                    "SHAP value": "{:+.4f}",
                    "Absolute share (%)": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Absolute share is the fraction of total absolute SHAP magnitude "
                "for this specific prediction; it is not global feature importance."
            )
    except (DeploymentError, ValueError, RuntimeError) as exc:
        st.warning(f"Prediction completed, but an honest SHAP explanation is unavailable: {exc}")
    except Exception as exc:
        st.warning(
            "Prediction completed, but SHAP could not be drawn with these library "
            f"versions. Details: {exc}"
        )

st.markdown(
    """
    <div class="footer">
    <b>Author:</b> Yifan Sun, M.D., Ph.D.<br>
    <b>Affiliation:</b> Guigang City People's Hospital, Guangxi, China<br>
    <i>Powered by Streamlit | Developed for clinical research and precision medicine.</i>
    </div>
    """,
    unsafe_allow_html=True,
)
