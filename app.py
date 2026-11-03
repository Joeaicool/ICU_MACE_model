
"""
===========================================================
ICU MACE Risk Prediction — Streamlit Application
===========================================================

Models:
    XGB_GA.pkl
    RF_BO_best.pkl

Features:
    Temperature, AST, Cr, ALP, Uric Acid

Original preprocessing:
    KNNImputer(n_neighbors=5)
    StandardScaler()
    Boruta feature selection

Training random seed: 23

Risk classification:
    Probability >= 0.50 -> High Risk
    Probability <  0.50 -> Low Risk

The 0.50 cutoff is a model-based classification threshold,
not an independently validated clinical MACE risk threshold.

IMAGE SETTINGS:
    The image is displayed ONLY at the bottom of the page.
    Image width = 150 pixels.

Research use only.
===========================================================
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

RANDOM_SEED = 23

# Original binary classification threshold
RISK_THRESHOLD = 0.50

FEATURES = [
    "Temperature",
    "AST",
    "Cr",
    "ALP",
    "Uric Acid",
]

MODEL_FILES = {
    "XGBoost (Genetic Algorithm)": "XGB_GA.pkl",
    "Random Forest (Bayesian)": "RF_BO_best.pkl",
}


# ============================================================
# 2. ORIGINAL MODEL FILE VERIFICATION
# ============================================================

MODEL_SHA256 = {
    "XGB_GA.pkl": (
        "8cff044c1105d77607c7cc803d1212023aa7e511fd573b0a64289682ea3d002a"
    ),
    "RF_BO_best.pkl": (
        "b7ad1c18b611d87a8de9de73de495a49224c87fa014563e53109c2783297e161"
    ),
}


# ============================================================
# 3. ORIGINAL TRAINING STANDARDIZATION PARAMETERS
# ============================================================

# Reconstructed from the original training dataset.
#
# Original training procedure:
# train_test_split(
#     test_size=0.3,
#     random_state=23,
#     stratify=y
# )
#
# KNNImputer(n_neighbors=5)
# StandardScaler()
#
# Feature order:
# Temperature, AST, Cr, ALP, Uric Acid

TRAIN_MEAN = np.array(
    [
        37.23268542199489,
        115.38900255754477,
        184.07263427109973,
        134.35780051150894,
        410.535294117647,
    ],
    dtype=np.float64,
)

TRAIN_SCALE = np.array(
    [
        1.1252469809105692,
        362.9121524778974,
        163.0057794559057,
        102.88511562493935,
        198.02199148675766,
    ],
    dtype=np.float64,
)


# ============================================================
# 4. INPUT INFORMATION
# ============================================================

# Descriptive percentiles from original training data.
# These are NOT clinical reference ranges.

TRAIN_P05_P95 = {
    "Temperature": (36.0, 39.5),
    "AST": (12.0, 304.4),
    "Cr": (50.0, 552.9),
    "ALP": (50.0, 344.9),
    "Uric Acid": (148.2, 778.7),
}

INPUT_STEPS = {
    "Temperature": 0.1,
    "AST": 1.0,
    "Cr": 1.0,
    "ALP": 1.0,
    "Uric Acid": 1.0,
}


# ============================================================
# 5. FOOTER IMAGE SETTINGS
# ============================================================

# 图片只在网页底部显示
# 不在网页顶部显示
# 不在侧边栏显示

FOOTER_IMAGE_WIDTH = 150

# 自动寻找图片文件
IMAGE_CANDIDATES = [
    "images.png",
    "images.jpeg",
    "images.jpg",
    "image.png",
    "image.jpeg",
    "logo.png",
]


def find_footer_image():

    for filename in IMAGE_CANDIDATES:

        image_path = APP_DIR / filename

        if image_path.is_file():
            return image_path

    return None


FOOTER_IMAGE_PATH = find_footer_image()


# ============================================================
# 6. CUSTOM ERROR
# ============================================================

class PredictorError(RuntimeError):
    pass


# ============================================================
# 7. CHECK MODEL HASH
# ============================================================

def calculate_sha256(file_path):

    digest = hashlib.sha256()

    with open(file_path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 8. LOAD ORIGINAL MODEL
# ============================================================

@lru_cache(maxsize=2)
def load_model(model_filename):

    if model_filename not in MODEL_FILES.values():

        raise PredictorError(
            "Unexpected model filename."
        )

    model_path = APP_DIR / model_filename

    if not model_path.is_file():

        raise PredictorError(
            f"Model file not found: {model_filename}"
        )

    # Verify original model file

    actual_hash = calculate_sha256(
        model_path
    )

    expected_hash = MODEL_SHA256[
        model_filename
    ]

    if actual_hash != expected_hash:

        raise PredictorError(
            f"{model_filename} does not match "
            "the original verified model. "
            "Prediction has been blocked."
        )

    try:

        model = joblib.load(model_path)

    except Exception as e:

        raise PredictorError(
            f"Model loading failed: {e}"
        )

    if not hasattr(model, "predict_proba"):

        raise PredictorError(
            "Model does not support probability prediction."
        )

    # Check features

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

    elif hasattr(model, "get_booster"):

        model_features = (
            model.get_booster().feature_names
        )

    else:

        raise PredictorError(
            "Model feature names are unavailable."
        )

    if model_features is None:

        raise PredictorError(
            "Unable to verify model feature names."
        )

    if list(model_features) != FEATURES:

        raise PredictorError(
            "Model feature order mismatch. "
            f"Expected: {FEATURES}; "
            f"Actual: {model_features}"
        )

    if getattr(model, "n_features_in_", None) != 5:

        raise PredictorError(
            "Model feature count mismatch."
        )

    if list(model.classes_) != [0, 1]:

        raise PredictorError(
            "Unexpected classification labels."
        )

    return model


# ============================================================
# 9. PATIENT PREPROCESSING
# ============================================================

def preprocess_patient(patient_values):

    """
    Apply original StandardScaler transformation.

    The five selected biomarkers must have
    observed values.

    KNNImputer does not modify observed values,
    so complete inputs can be standardized using
    the original fitted means and scales.
    """

    if set(patient_values.keys()) != set(FEATURES):

        raise PredictorError(
            "All five biomarkers are required."
        )

    try:

        raw_array = np.array(
            [
                [
                    float(patient_values[f])
                    for f in FEATURES
                ]
            ],
            dtype=np.float64,
        )

    except (TypeError, ValueError):

        raise PredictorError(
            "Biomarker inputs must be numeric."
        )

    if not np.isfinite(raw_array).all():

        raise PredictorError(
            "Inputs must contain finite values."
        )

    if np.any(raw_array <= 0):

        raise PredictorError(
            "All five biomarkers must be greater "
            "than zero. Do not use zero for missing data."
        )

    if not np.isfinite(TRAIN_MEAN).all():

        raise PredictorError(
            "Invalid training means."
        )

    if not np.isfinite(TRAIN_SCALE).all():

        raise PredictorError(
            "Invalid training standardization parameters."
        )

    if np.any(TRAIN_SCALE <= 0):

        raise PredictorError(
            "Invalid training scales."
        )

    raw_df = pd.DataFrame(
        raw_array,
        columns=FEATURES,
    )

    # Original StandardScaler transformation

    transformed_array = (
        raw_array - TRAIN_MEAN
    ) / TRAIN_SCALE

    transformed_df = pd.DataFrame(
        transformed_array,
        columns=FEATURES,
    )

    return raw_df, transformed_df


# ============================================================
# 10. MODEL PREDICTION
# ============================================================

def predict_patient(model, transformed_df):

    if list(transformed_df.columns) != FEATURES:

        raise PredictorError(
            "Feature order mismatch during prediction."
        )

    try:

        probabilities = np.asarray(
            model.predict_proba(
                transformed_df
            ),
            dtype=np.float64,
        )

    except Exception as e:

        raise PredictorError(
            f"Model prediction failed: {e}"
        )

    if probabilities.shape != (1, 2):

        raise PredictorError(
            "Unexpected probability output."
        )

    if not np.isfinite(probabilities).all():

        raise PredictorError(
            "The model returned invalid probabilities."
        )

    # Positive class MACE = 1

    positive_index = list(
        model.classes_
    ).index(1)

    risk_probability = float(
        probabilities[0, positive_index]
    )

    if not 0 <= risk_probability <= 1:

        raise PredictorError(
            "Prediction score is outside [0, 1]."
        )

    # ========================================================
    # HIGH / LOW RISK CLASSIFICATION
    # ========================================================

    if risk_probability >= RISK_THRESHOLD:

        risk_category = "High Risk"

        risk_message = "⚠️ High Risk Detected"

        risk_color = "#DC3545"

        risk_description = (
            "The machine learning model classifies "
            "this patient as higher risk for MACE "
            "according to its default 50% "
            "decision threshold."
        )

    else:

        risk_category = "Low Risk"

        risk_message = "✅ Low Risk Detected"

        risk_color = "#198754"

        risk_description = (
            "The machine learning model classifies "
            "this patient as lower risk for MACE "
            "according to its default 50% "
            "decision threshold."
        )

    return {
        "probability": risk_probability,
        "percentage": risk_probability * 100,
        "category": risk_category,
        "message": risk_message,
        "color": risk_color,
        "description": risk_description,
    }


# ============================================================
# 11. SHAP EXPLANATION
# ============================================================

def get_shap_explanation(
    model,
    transformed_df,
    raw_df,
    predicted_probability,
):

    import shap

    from scipy.special import expit

    is_xgb = hasattr(
        model,
        "get_booster"
    )

    try:

        # Tree SHAP
        # Do not modify XGBoost model parameters

        explainer = shap.TreeExplainer(model)

        explanation = explainer(
            transformed_df
        )

        values = np.asarray(
            explanation.values,
            dtype=float,
        )

        base_values = np.asarray(
            explanation.base_values,
            dtype=float,
        )

        # Random Forest binary output

        if values.shape == (1, 5, 2):

            shap_values = values[0, :, 1]

            base_value = float(
                base_values.reshape(1, -1)[0, 1]
            )

            shap_units = "probability"

        # XGBoost binary output

        elif values.shape == (1, 5):

            shap_values = values[0]

            base_value = float(
                base_values.reshape(-1)[0]
            )

            if is_xgb:
                shap_units = "log-odds"
            else:
                shap_units = "probability"

        else:

            raise ValueError(
                f"Unsupported SHAP shape: {values.shape}"
            )

        shap_sum = (
            base_value + shap_values.sum()
        )

        if shap_units == "log-odds":

            reconstructed_probability = float(
                expit(shap_sum)
            )

        else:

            reconstructed_probability = float(
                shap_sum
            )

        # Additivity check

        if not np.isclose(
            reconstructed_probability,
            predicted_probability,
            rtol=5e-4,
            atol=5e-4,
        ):

            raise ValueError(
                "Tree SHAP additivity verification failed."
            )

        reference_name = (
            "TreeExplainer model reference"
        )

    except Exception as tree_error:

        # ====================================================
        # SHAP FALLBACK
        # ====================================================

        def blackbox_predict(x):

            x = np.asarray(
                x,
                dtype=float,
            )

            df = pd.DataFrame(
                x,
                columns=FEATURES,
            )

            return model.predict_proba(
                df
            )[:, 1]

        try:

            background = np.zeros(
                (1, len(FEATURES)),
                dtype=float,
            )

            kernel_explainer = shap.KernelExplainer(
                blackbox_predict,
                background,
            )

            kernel_result = (
                kernel_explainer.shap_values(
                    transformed_df.to_numpy(),
                    nsamples=128,
                    silent=True,
                )
            )

            kernel_values = np.asarray(
                kernel_result,
                dtype=float,
            )

            if kernel_values.shape == (1, 5):

                shap_values = kernel_values[0]

            elif kernel_values.shape == (5,):

                shap_values = kernel_values

            else:

                raise ValueError(
                    "Unexpected Kernel SHAP output."
                )

            base_value = float(
                np.asarray(
                    kernel_explainer.expected_value
                ).reshape(-1)[0]
            )

            shap_units = "probability"

            reference_name = (
                "Synthetic training-mean reference"
            )

            reconstructed_probability = float(
                base_value + shap_values.sum()
            )

            if not np.isclose(
                reconstructed_probability,
                predicted_probability,
                rtol=1e-3,
                atol=1e-3,
            ):

                raise ValueError(
                    "Kernel SHAP additivity verification failed."
                )

        except Exception as kernel_error:

            raise PredictorError(
                "SHAP explanation failed. "
                f"Tree SHAP: {tree_error}. "
                f"Kernel SHAP: {kernel_error}"
            )

    shap_exp = shap.Explanation(
        values=np.asarray(
            shap_values,
            dtype=float,
        ),
        base_values=float(
            base_value
        ),
        data=raw_df.iloc[0].to_numpy(
            dtype=float
        ),
        feature_names=FEATURES,
    )

    return (
        shap_exp,
        shap_units,
        reference_name,
    )


# ============================================================
# 12. STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=(
        "AI Clinical Risk Predictor | Precision Medicine"
    ),
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 13. CSS DESIGN
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #F8F9FA;
}

.title-box {
    background: linear-gradient(
        135deg,
        #0A2540,
        #1750A1
    );

    padding: 2rem 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}

.title-box h1 {
    color: white;
    font-size: 2.2rem;
    font-weight: 750;
    margin: 0;
}

.title-box p {
    color: #EDF4FF;
    font-size: 1.05rem;
    margin-top: 12px;
}

.clinical-note {
    background-color: #EBF4FA;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    border-left: 6px solid #1750A1;
    color: #0A2540;
    margin-bottom: 2rem;
}

.risk-high {
    background-color: #FDECEC;
    border-left: 7px solid #DC3545;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.risk-high h2 {
    color: #B42318;
    font-size: 1.8rem;
    margin: 0 0 10px 0;
}

.risk-low {
    background-color: #EAF7EF;
    border-left: 7px solid #198754;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.risk-low h2 {
    color: #146C43;
    font-size: 1.8rem;
    margin: 0 0 10px 0;
}

.risk-description {
    color: #334155;
    font-size: 1rem;
    line-height: 1.6;
}

.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #E2E8F0;
    text-align: center;
    color: #64748B;
    font-size: 0.9rem;
}

.footer-image-title {
    text-align: center;
    color: #94A3B8;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 14. SIDEBAR - MODEL SELECTION
# ============================================================

st.sidebar.markdown(
    "## ⚙️ Model Selection"
)

st.sidebar.markdown(
    "Select the AI architecture based "
    "on clinical research needs:"
)

selected_model_name = st.sidebar.radio(
    "Available models",
    list(MODEL_FILES.keys()),
    label_visibility="collapsed",
)

MODEL_FILE = MODEL_FILES[
    selected_model_name
]

st.sidebar.markdown("---")

st.sidebar.info(
    f"**Currently Active:**\n\n"
    f"{selected_model_name}\n\n"
    f"*File: `{MODEL_FILE}`*"
)

st.sidebar.markdown(
    "**Model Classification Threshold:** 50%"
)

st.sidebar.caption(
    "Scores of 50% or higher are classified "
    "as High Risk. Scores below 50% are "
    "classified as Low Risk. "
    "This is a model-based threshold, not "
    "a validated clinical decision threshold."
)


# ============================================================
# 15. MAIN PAGE HEADER
# ============================================================

# NOTE:
# No image is displayed here.
# Image display has been moved to the bottom.

st.markdown(
    """
<div class="title-box">

<h1>
Advanced Machine Learning Model for Clinical Risk Prediction
</h1>

<p>
Explainable Artificial Intelligence (XAI)
for Personalized Patient Assessment
</p>

</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="clinical-note">

<b>Objective:</b>
This tool leverages optimized machine learning
algorithms (XGBoost and Random Forest) to predict
individual MACE model scores using five
clinical biomarkers.

<br><br>

<b>Clinical Risk Classification:</b>
Patients are categorized as High Risk or Low Risk
using the model's default 50% decision threshold.

<br><br>

<b>Important Notice:</b>
This classification is intended for clinical
research and is not an independently validated
clinical risk threshold.
Final medical decisions must be made by qualified
healthcare professionals.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 16. LOAD MODEL
# ============================================================

try:

    model = load_model(
        MODEL_FILE
    )

except PredictorError as e:

    st.error(
        f"⚠️ Model loading or verification failed: {e}"
    )

    st.stop()

st.success(
    "Verified original classifier and "
    "training-fitted preprocessing loaded."
)


# ============================================================
# 17. PATIENT INPUT FORM
# ============================================================

st.markdown(
    "### 📝 Step 1: Patient Biomarker Input"
)

st.caption(
    "Enter all five original laboratory measurements. "
    "Please use the same units as the original "
    "training dataset."
)

patient_values = {}

with st.form(
    "patient_input_form",
    border=True,
):

    col1, col2, col3 = st.columns(3)

    columns = [
        col1,
        col2,
        col3,
    ]

    for idx, feature in enumerate(FEATURES):

        low, high = TRAIN_P05_P95[feature]

        with columns[idx % 3]:

            patient_values[feature] = (
                st.number_input(
                    label=feature,
                    value=None,
                    step=INPUT_STEPS[feature],
                    format="%.2f",
                    key=f"input_{feature}",
                    help=(
                        "Training-data 5th–95th "
                        f"percentile: {low:g}–{high:g}. "
                        "This is not a clinical "
                        "reference range."
                    ),
                )
            )

    predict_btn = st.form_submit_button(
        label=(
            "🚀 Run Assessment with "
            f"{selected_model_name.split(' ')[0]}"
        ),
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 18. RUN PREDICTION
# ============================================================

if predict_btn:

    try:

        if any(
            v is None
            for v in patient_values.values()
        ):

            raise PredictorError(
                "Please enter all five biomarkers."
            )

        raw_df, standardized_df = (
            preprocess_patient(
                patient_values
            )
        )

        result = predict_patient(
            model,
            standardized_df,
        )

    except PredictorError as e:

        st.error(
            f"Assessment failed: {e}"
        )

        st.stop()


    # ========================================================
    # 19. INPUT DISTRIBUTION CHECK
    # ========================================================

    unusual_features = []

    for feature in FEATURES:

        patient_value = float(
            raw_df.at[0, feature]
        )

        p05, p95 = TRAIN_P05_P95[feature]

        if (
            patient_value < p05
            or patient_value > p95
        ):

            unusual_features.append(
                feature
            )

    if unusual_features:

        st.warning(
            "The following biomarkers are outside "
            "the central 90% of the training "
            "observations: "
            f"**{', '.join(unusual_features)}**. "
            "Please verify laboratory units and "
            "interpret the result cautiously. "
            "These percentiles are not medical "
            "normal ranges."
        )

    if np.max(
        np.abs(
            standardized_df.to_numpy()
        )
    ) > 6:

        st.warning(
            "One or more measurements are "
            "more than six training standard "
            "deviations from the mean. "
            "Please verify the entered values."
        )


    # ========================================================
    # 20. RISK RESULTS
    # ========================================================

    st.markdown(
        "### 📊 Step 2: Prediction Results"
    )

    result_col1, result_col2 = st.columns(
        [1.15, 1],
        gap="large",
    )

    with result_col1:

        st.markdown(
            "#### 🩺 Clinical Risk Classification"
        )

        # HIGH RISK

        if result["category"] == "High Risk":

            st.markdown(
                f"""
<div class="risk-high">

<h2>
⚠️ High Risk Detected
</h2>

<div class="risk-description">

<b>Classification:</b>
High Risk

<br><br>

{result["description"]}

</div>

</div>
""",
                unsafe_allow_html=True,
            )

        # LOW RISK

        else:

            st.markdown(
                f"""
<div class="risk-low">

<h2>
✅ Low Risk Detected
</h2>

<div class="risk-description">

<b>Classification:</b>
Low Risk

<br><br>

{result["description"]}

</div>

</div>
""",
                unsafe_allow_html=True,
            )

        st.metric(
            label="Calculated MACE Model Score",
            value=f"{result['percentage']:.4f}%",
        )

        st.write(
            f"**Model Classification:** "
            f"{result['category']}"
        )

        st.write(
            "**Classification Threshold:** 50%"
        )

        st.caption(
            "This score is the model's predicted "
            "positive-class output. The classification "
            "uses the original binary-model threshold "
            "and is not a clinically validated "
            "MACE risk stratification rule."
        )

    with result_col2:

        # ====================================================
        # RISK GAUGE
        # ====================================================

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["percentage"],
                number={
                    "suffix": "%",
                    "valueformat": ".2f",
                    "font": {
                        "size": 40,
                        "color": "#0A2540",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                    },
                    "bar": {
                        "color": result["color"],
                    },
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#D1D5DB",
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "rgba(25,135,84,0.14)",
                        },
                        {
                            "range": [50, 100],
                            "color": "rgba(220,53,69,0.14)",
                        },
                    ],
                    "threshold": {
                        "line": {
                            "color": "#0A2540",
                            "width": 4,
                        },
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )

        gauge.update_layout(
            height=290,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20,
            ),
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

        if result["category"] == "High Risk":

            st.error(
                "⚠️ Model-Based Classification: HIGH RISK"
            )

        else:

            st.success(
                "✅ Model-Based Classification: LOW RISK"
            )

        st.caption(
            "Green: model score below 50%. "
            "Red: model score at or above 50%. "
            "The scale represents model classification, "
            "not a clinically established risk scale."
        )


    # ========================================================
    # 21. CLINICAL INTERPRETATION
    # ========================================================

    st.markdown(
        "#### 📋 Risk Interpretation"
    )

    if result["category"] == "High Risk":

        st.warning(
            "**Higher Model-Predicted MACE Risk:** "
            "The model assigns this patient to its "
            "positive classification category. "
            "This finding warrants clinical review "
            "in the context of the complete patient "
            "history, examinations, and laboratory "
            "results. It does not independently "
            "establish that MACE will occur."
        )

    else:

        st.info(
            "**Lower Model-Predicted MACE Risk:** "
            "The model assigns this patient to its "
            "negative classification category. "
            "This result does not exclude MACE or "
            "replace clinical evaluation."
        )


    # ========================================================
    # 22. INPUT AUDIT
    # ========================================================

    with st.expander(
        "🔎 Prediction Audit: "
        "Raw Values and Model Inputs"
    ):

        audit_df = pd.DataFrame(
            {
                "Biomarker": FEATURES,
                "Raw Input": raw_df.iloc[0].to_numpy(),
                "Training Mean": TRAIN_MEAN,
                "Training Scale": TRAIN_SCALE,
                "Standardized Input": (
                    standardized_df.iloc[0].to_numpy()
                ),
            }
        )

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Standardized Input = "
            "(Raw Input − Training Mean) "
            "/ Training Scale"
        )


    # ========================================================
    # 23. SHAP EXPLAINABILITY
    # ========================================================

    st.markdown(
        "### 🔍 Step 3: AI Explainability (SHAP Analysis)"
    )

    st.write(
        "The plots below explain how the "
        "five biomarkers contribute to the "
        "individual model prediction."
    )

    try:

        with st.spinner(
            "Calculating personalized SHAP values..."
        ):

            (
                shap_explanation,
                shap_units,
                reference_name,
            ) = get_shap_explanation(
                model,
                standardized_df,
                raw_df,
                result["probability"],
            )

        st.caption(
            f"SHAP units: {shap_units}. "
            f"Reference: {reference_name}. "
            "SHAP contributions describe model "
            "behavior, not causal medical effects."
        )

        shap_col1, shap_col2 = st.columns(
            [1.1, 1],
            gap="large",
        )

        # ====================================================
        # SHAP WATERFALL
        # ====================================================

        with shap_col1:

            st.markdown(
                "**Individual SHAP Waterfall Plot**"
            )

            import shap

            plt.close("all")

            shap.plots.waterfall(
                shap_explanation,
                max_display=5,
                show=False,
            )

            fig = plt.gcf()

            fig.set_size_inches(
                7.5,
                5,
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)


        # ====================================================
        # SHAP CONTRIBUTION TABLE
        # ====================================================

        with shap_col2:

            st.markdown(
                "**Feature Contribution Ranking**"
            )

            shap_values = np.asarray(
                shap_explanation.values,
                dtype=float,
            )

            abs_values = np.abs(
                shap_values
            )

            total = abs_values.sum()

            if total > 0:

                percentages = (
                    abs_values / total
                ) * 100

            else:

                percentages = np.zeros_like(
                    abs_values
                )

            effects = []

            for value in shap_values:

                if value > 0:

                    effects.append(
                        "⬆️ Increased Model Output"
                    )

                elif value < 0:

                    effects.append(
                        "⬇️ Decreased Model Output"
                    )

                else:

                    effects.append(
                        "No Effect"
                    )

            contribution_df = pd.DataFrame(
                {
                    "Biomarker": FEATURES,
                    "Patient Value": raw_df.iloc[0].values,
                    "Effect": effects,
                    "SHAP Value": shap_values,
                    "Contribution Impact (%)": percentages,
                }
            )

            contribution_df = (
                contribution_df.sort_values(
                    "Contribution Impact (%)",
                    ascending=False,
                )
            )

            styled = (
                contribution_df.style.format(
                    {
                        "Patient Value": "{:.2f}",
                        "SHAP Value": "{:+.4f}",
                        "Contribution Impact (%)": "{:.1f}%",
                    }
                )
            )

            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Contribution Impact (%) represents "
                "the relative absolute SHAP contribution "
                "within this prediction. It is not the "
                "percentage of clinical risk attributable "
                "to a biomarker."
            )

    except Exception as e:

        st.warning(
            "The MACE prediction was completed "
            "successfully, but SHAP visualization "
            "could not be generated. "
            f"Technical details: {e}"
        )


    # ========================================================
    # 24. RESEARCH DISCLAIMER
    # ========================================================

    st.markdown("---")

    st.caption(
        "**Research Disclaimer:** "
        "This tool is a supplementary research "
        "decision-support application. Its High Risk "
        "and Low Risk categories are determined by "
        "the model's default 0.50 classification "
        "threshold. This threshold has not been "
        "established as a clinically validated MACE "
        "decision threshold. Predictions should not "
        "be used as the sole basis for medical decisions."
    )


# ============================================================
# 25. FOOTER — AUTHOR INFORMATION
# ============================================================

st.markdown(
    """
<div class="footer">

<b>Author:</b>
Yifan Sun, M.D., Ph.D.

<br>

<b>Affiliation:</b>
Guigang City People's Hospital,
Guangxi, China

<br><br>

<i>
Powered by Streamlit |
Developed for Clinical Research
and Precision Medicine.
</i>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 26. SMALL FOOTER IMAGE — FINAL POSITION
# ============================================================

# 图片现在只在整个网页最底部出现一次。
#
# 位置：
#   作者信息之后
#   网页最底部
#
# 大小：
#   150 px
#
# 对齐：
#   居中
#
# 如需更小：
#   修改 FOOTER_IMAGE_WIDTH = 100
#
# 如需更大：
#   修改 FOOTER_IMAGE_WIDTH = 200

if FOOTER_IMAGE_PATH is not None:

    st.markdown(
        "<div style='height:15px'></div>",
        unsafe_allow_html=True,
    )

    left_space, image_center, right_space = st.columns(
        [3, 1, 3]
    )

    with image_center:

        st.image(
            str(FOOTER_IMAGE_PATH),
            width=FOOTER_IMAGE_WIDTH,
        )

    st.markdown(
        """
<div class="footer-image-title">
Clinical Research · Precision Medicine
</div>
""",
        unsafe_allow_html=True,
    )
