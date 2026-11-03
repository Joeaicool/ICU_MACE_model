
"""
ICU MACE Risk Prediction — Streamlit Application

Original models:
    XGB_GA.pkl
    RF_BO_best.pkl

Original training dataset:
    macedata_filtered.xlsx

Training pipeline:
    1. Remove excluded and high-missingness variables
    2. Stratified train/test split, random_state=23
    3. KNNImputer(n_neighbors=5)
    4. StandardScaler()
    5. Boruta feature selection
    6. XGBoost / Random Forest prediction

The preprocessing parameters below have been reconstructed
from the original training dataset.

Only five fully observed biomarkers are required.
The original Excel file is NOT needed for online deployment.

This is a research demonstration and not a clinically
validated, calibrated risk calculator.
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
# 1. GLOBAL SETTINGS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

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
# 2. MODEL FILE VERIFICATION
# ============================================================

# SHA256 values of the two original uploaded models.
# Prevents accidental use of different model versions.

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

# Reconstructed using the original training code:
#
# random_state = 23
# test_size = 0.3
# stratify = y
# KNNImputer(n_neighbors=5)
# StandardScaler()
#
# These are fitted training-data parameters.
# They are NOT clinical reference values.

TRAIN_MEAN = np.array(
    [
        37.23268542199489,       # Temperature
        115.38900255754477,      # AST
        184.07263427109973,      # Cr
        134.35780051150894,      # ALP
        410.535294117647,        # Uric Acid
    ],
    dtype=np.float64,
)

TRAIN_SCALE = np.array(
    [
        1.1252469809105692,      # Temperature
        362.9121524778974,       # AST
        163.0057794559057,      # Cr
        102.88511562493935,     # ALP
        198.02199148675766,     # Uric Acid
    ],
    dtype=np.float64,
)


# ============================================================
# 4. INPUT DISTRIBUTION INFORMATION
# ============================================================

# Descriptive training-data percentiles.
# These are NOT medical normal ranges.

TRAIN_OBSERVED_P05_P95 = {
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

# The source workbook does not document laboratory units.
# All inputs must use the same units as the training dataset.

# No independently validated clinical cutoff was supplied.
# Therefore, this app reports continuous model scores rather
# than assigning arbitrary high-risk / low-risk categories.


# ============================================================
# 5. CUSTOM ERROR CLASS
# ============================================================

class PredictorError(RuntimeError):
    """Stop prediction when model or input validation fails."""
    pass


# ============================================================
# 6. CHECK MODEL FILE HASH
# ============================================================

def file_sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as source:

        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 7. LOAD AND VERIFY MODEL
# ============================================================

@lru_cache(maxsize=2)
def load_verified_model(filename: str):

    if filename not in MODEL_SHA256:

        raise PredictorError(
            f"Unexpected model filename: {filename}"
        )

    model_path = APP_DIR / filename

    if not model_path.is_file():

        raise PredictorError(
            f"Model file {filename} was not found. "
            "Keep the original model file next to app.py."
        )

    actual_hash = file_sha256(model_path)
    expected_hash = MODEL_SHA256[filename]

    if actual_hash != expected_hash:

        raise PredictorError(
            f"The uploaded {filename} does not match the "
            "original verified model. Prediction has been "
            "blocked to avoid a preprocessing mismatch."
        )

    try:

        model = joblib.load(model_path)

    except Exception as exc:

        raise PredictorError(
            f"Unable to load {filename}. "
            f"Please check package compatibility. {exc}"
        ) from exc

    if not callable(getattr(model, "predict_proba", None)):

        raise PredictorError(
            "The model does not support probability prediction."
        )

    # Verify feature names.

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        if callable(getattr(model, "get_booster", None)):

            feature_names = (
                model.get_booster().feature_names
            )

    if feature_names is None:

        raise PredictorError(
            "The original feature names could not be verified."
        )

    model_features = [
        str(feature) for feature in feature_names
    ]

    if model_features != FEATURES:

        raise PredictorError(
            "Model feature order mismatch. "
            f"Expected {FEATURES}; "
            f"received {model_features}."
        )

    if getattr(model, "n_features_in_", None) != 5:

        raise PredictorError(
            "This model does not contain five input features."
        )

    model_classes = list(
        getattr(model, "classes_", [])
    )

    if model_classes != [0, 1]:

        raise PredictorError(
            "The model positive class is not MACE = 1."
        )

    return model


# ============================================================
# 8. PATIENT DATA PREPROCESSING
# ============================================================

def transform_patient(raw_values):

    """
    Convert original clinical values into the standardized
    feature values used during model training.

    The original pipeline fitted KNNImputer on all training
    predictors before StandardScaler.

    Here, all five selected biomarkers must be observed.
    KNNImputer does not alter an observed biomarker value,
    so using the fitted mean and scale of these five columns
    reproduces the corresponding original model inputs.
    """

    if set(raw_values.keys()) != set(FEATURES):

        raise PredictorError(
            "All five biomarkers must be supplied."
        )

    if not np.isfinite(TRAIN_MEAN).all():

        raise PredictorError(
            "Invalid training means."
        )

    if not np.isfinite(TRAIN_SCALE).all():

        raise PredictorError(
            "Invalid training standard deviations."
        )

    if np.any(TRAIN_SCALE <= 0):

        raise PredictorError(
            "The training scaler contains invalid values."
        )

    try:

        values = np.array(
            [[float(raw_values[f]) for f in FEATURES]],
            dtype=np.float64,
        )

    except (ValueError, TypeError) as exc:

        raise PredictorError(
            "Biomarker values must be numeric."
        ) from exc

    if not np.isfinite(values).all():

        raise PredictorError(
            "Missing or non-finite values are not allowed."
        )

    if np.any(values <= 0):

        raise PredictorError(
            "All biomarkers must be greater than zero. "
            "Do not use zero to represent a missing value."
        )

    raw_df = pd.DataFrame(
        values,
        columns=FEATURES,
    )

    # Original training transformation:
    #
    # X_standardized = (X_original - mean) / scale

    standardized_values = (
        values - TRAIN_MEAN
    ) / TRAIN_SCALE

    if not np.isfinite(standardized_values).all():

        raise PredictorError(
            "Standardization produced an invalid value."
        )

    standardized_df = pd.DataFrame(
        standardized_values,
        columns=FEATURES,
    )

    return raw_df, standardized_df


# ============================================================
# 9. PREDICT MACE POSITIVE-CLASS SCORE
# ============================================================

def positive_probability(model, standardized_df):

    if list(standardized_df.columns) != FEATURES:

        raise PredictorError(
            "Incorrect feature order."
        )

    if standardized_df.shape != (1, 5):

        raise PredictorError(
            "Incorrect model input dimensions."
        )

    try:

        probabilities = np.asarray(
            model.predict_proba(standardized_df),
            dtype=np.float64,
        )

    except Exception as exc:

        raise PredictorError(
            f"Prediction failed: {exc}"
        ) from exc

    if probabilities.shape != (1, 2):

        raise PredictorError(
            "Invalid probability output shape."
        )

    if not np.isfinite(probabilities).all():

        raise PredictorError(
            "Model returned non-finite probability values."
        )

    # Both verified models have classes_ = [0, 1].
    # Column 1 represents MACE = 1.

    probability = float(
        probabilities[0, 1]
    )

    if not 0 <= probability <= 1:

        raise PredictorError(
            "The model score is outside the valid range."
        )

    return probability


# ============================================================
# 10. SHAP EXPLANATION
# ============================================================

def explain_patient(
    model,
    standardized_df,
    raw_df,
    model_score,
):

    import shap
    from scipy.special import expit

    is_xgboost = callable(
        getattr(model, "get_booster", None)
    )

    try:

        # Preferred Tree SHAP explanation.

        explainer = shap.TreeExplainer(model)

        explanation = explainer(
            standardized_df
        )

        all_values = np.asarray(
            explanation.values,
            dtype=float,
        )

        all_base_values = np.asarray(
            explanation.base_values,
            dtype=float,
        )

        # RF binary classification may return:
        #
        # (samples, features, classes)

        if all_values.shape == (1, 5, 2):

            shap_values = all_values[0, :, 1]

            base_value = float(
                all_base_values.reshape(1, -1)[0, 1]
            )

            shap_units = "probability"

        # XGBoost commonly returns:
        #
        # (samples, features)

        elif all_values.shape == (1, 5):

            shap_values = all_values[0]

            base_value = float(
                all_base_values.reshape(-1)[0]
            )

            shap_units = (
                "log-odds"
                if is_xgboost
                else "probability"
            )

        else:

            raise ValueError(
                f"Unexpected SHAP dimensions: "
                f"{all_values.shape}"
            )

        # Check SHAP additivity.

        shap_sum = (
            base_value + shap_values.sum()
        )

        if shap_units == "log-odds":

            reconstructed_score = float(
                expit(shap_sum)
            )

        else:

            reconstructed_score = float(
                shap_sum
            )

        if not np.isclose(
            reconstructed_score,
            model_score,
            rtol=5e-4,
            atol=5e-4,
        ):

            raise ValueError(
                "Tree SHAP values do not reproduce "
                "the model prediction."
            )

        reference_name = (
            "TreeExplainer model reference"
        )

    except Exception as tree_error:

        # Fallback for SHAP and XGBoost version issues.
        # The trained model is never modified.
        #
        # Background = mean patient in standardized space.
        # This is a synthetic reference, not a real patient.

        def predict_standardized(zscores):

            x = np.asarray(
                zscores,
                dtype=float,
            )

            frame = pd.DataFrame(
                x,
                columns=FEATURES,
            )

            return model.predict_proba(
                frame
            )[:, 1]

        try:

            background = np.zeros(
                (1, len(FEATURES)),
                dtype=np.float64,
            )

            kernel_explainer = shap.KernelExplainer(
                predict_standardized,
                background,
            )

            kernel_result = (
                kernel_explainer.shap_values(
                    standardized_df.to_numpy(),
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
                    "Unexpected Kernel SHAP shape: "
                    f"{kernel_values.shape}"
                )

            base_value = float(
                np.asarray(
                    kernel_explainer.expected_value
                ).reshape(-1)[0]
            )

            shap_units = "probability"

            reference_name = (
                "Synthetic training-mean reference "
                "(all standardized inputs = 0)"
            )

            reconstructed_score = (
                base_value + shap_values.sum()
            )

            if not np.isclose(
                reconstructed_score,
                model_score,
                rtol=1e-3,
                atol=1e-3,
            ):

                raise ValueError(
                    "Kernel SHAP additivity check failed."
                )

        except Exception as kernel_error:

            raise PredictorError(
                "SHAP is unavailable because of package "
                "compatibility. "
                f"TreeSHAP: {tree_error}; "
                f"KernelSHAP: {kernel_error}"
            ) from kernel_error

    # Waterfall chart explanation.
    # Show original clinical values as feature data.

    shap_explanation = shap.Explanation(
        values=np.asarray(
            shap_values,
            dtype=float,
        ),
        base_values=float(base_value),
        data=raw_df.iloc[0].to_numpy(
            dtype=float
        ),
        feature_names=FEATURES,
    )

    return (
        shap_explanation,
        shap_units,
        reference_name,
    )


# ============================================================
# 11. STREAMLIT PAGE CONFIGURATION
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
# 12. PAGE STYLING
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
    border-radius: 12px;
    padding: 1.9rem 1.3rem;
    color: white;
    text-align: center;
    margin-bottom: 1.35rem;
}

.title-box h1 {
    color: white;
    font-size: 2.05rem;
    margin: 0;
    font-weight: 750;
}

.title-box p {
    color: #EDF4FF;
    font-size: 1.05rem;
    margin: .65rem 0 0;
}

.clinical-note {
    padding: 1rem 1.3rem;
    border-left: 5px solid #1750A1;
    background: #EAF3FB;
    border-radius: 8px;
    color: #0A2540;
    margin-bottom: 1.4rem;
}

.footer {
    margin-top: 2.8rem;
    padding-top: 1.1rem;
    border-top: 1px solid #E2E8F0;
    color: #64748B;
    text-align: center;
    font-size: .9rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 13. SIDEBAR MODEL SELECTION
# ============================================================

st.sidebar.markdown(
    "## ⚙️ Model Selection"
)

st.sidebar.caption(
    "Select the AI architecture used "
    "for this research assessment."
)

selected_model_name = st.sidebar.radio(
    "AI architecture",
    list(MODEL_FILES.keys()),
    label_visibility="collapsed",
)

MODEL_FILE = MODEL_FILES[
    selected_model_name
]

st.sidebar.info(
    f"**Currently Active:**\n\n"
    f"{selected_model_name}\n\n"
    f"*File: `{MODEL_FILE}`*"
)

st.sidebar.caption(
    "Enter clinical biomarker values using the same "
    "laboratory measurement units as the "
    "original training dataset."
)


# ============================================================
# 14. OPTIONAL MEDICAL IMAGE
# ============================================================

banner_path = APP_DIR / "images.jpeg"

if banner_path.is_file():

    st.sidebar.image(
        str(banner_path),
        use_container_width=True,
    )


# ============================================================
# 15. HEADER
# ============================================================

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

<div class="clinical-note">

<b>Objective:</b>
This research tool uses optimized XGBoost and
Random Forest models to calculate individual
MACE prediction scores based on five clinical biomarkers.

<br>

<b>Clinical disclaimer:</b>
The displayed model score is not necessarily
a calibrated event probability.
This tool is intended for research and
should not be used as a standalone basis
for clinical decisions.

</div>
""",
    unsafe_allow_html=True,
)

if banner_path.is_file():

    st.image(
        str(banner_path),
        use_container_width=True,
    )


# ============================================================
# 16. LOAD SELECTED MODEL
# ============================================================

try:

    model = load_verified_model(
        MODEL_FILE
    )

except PredictorError as exc:

    st.error(
        f"⚠️ Model verification failed: {exc}"
    )

    st.stop()

st.success(
    "Verified original classifier and "
    "training-fitted preprocessing loaded."
)


# ============================================================
# 17. PATIENT BIOMARKER INPUT
# ============================================================

st.markdown(
    "### 📝 Step 1: Patient Biomarker Input"
)

st.caption(
    "Enter all five measured biomarker values. "
    "Use original, unstandardized laboratory values "
    "in the same units as the training dataset. "
    "Missing values are not supported."
)

patient_values = {}

with st.form(
    "patient_form",
    border=True,
):

    input_columns = st.columns(3)

    for index, feature in enumerate(FEATURES):

        lower, upper = (
            TRAIN_OBSERVED_P05_P95[feature]
        )

        with input_columns[index % 3]:

            patient_values[feature] = (
                st.number_input(
                    feature,
                    value=None,
                    step=INPUT_STEPS[feature],
                    format="%.2f",
                    key=f"biomarker_{feature}",
                    help=(
                        "Use the original training units. "
                        "Training 5th–95th percentile: "
                        f"{lower:g}–{upper:g}. "
                        "These are descriptive values, "
                        "not medical normal ranges."
                    ),
                )
            )

    submitted = st.form_submit_button(
        (
            "🚀 Run Assessment with "
            f"{selected_model_name.split(' ')[0]}"
        ),
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 18. PREDICTION
# ============================================================

if submitted:

    try:

        if any(
            value is None
            for value in patient_values.values()
        ):

            raise PredictorError(
                "Please enter all five biomarkers "
                "before running the assessment."
            )

        raw_input, standardized_input = (
            transform_patient(patient_values)
        )

        model_score = positive_probability(
            model,
            standardized_input,
        )

    except PredictorError as exc:

        st.error(
            f"Prediction could not be completed: {exc}"
        )

        st.stop()


    # ========================================================
    # 19. INPUT RANGE CHECK
    # ========================================================

    unusual_features = []

    for feature in FEATURES:

        value = float(
            raw_input.at[0, feature]
        )

        lower, upper = (
            TRAIN_OBSERVED_P05_P95[feature]
        )

        if value < lower or value > upper:

            unusual_features.append(
                feature
            )

    if unusual_features:

        st.warning(
            "The following biomarkers are outside "
            "the central 90% of the original training "
            "observations: "
            f"**{', '.join(unusual_features)}**. "
            "Please verify measurement units and "
            "interpret predictions cautiously. "
            "This is not a clinical normal-range alert."
        )

    maximum_zscore = np.max(
        np.abs(
            standardized_input.to_numpy()
        )
    )

    if maximum_zscore > 6:

        st.warning(
            "At least one biomarker is more than "
            "six training standard deviations from "
            "the training mean. "
            "Please verify the input values and units."
        )


    # ========================================================
    # 20. PREDICTION RESULTS
    # ========================================================

    st.markdown(
        "### 📊 Step 2: Prediction Results"
    )

    result_left, result_right = st.columns(
        [1.1, 1],
        gap="large",
    )

    with result_left:

        st.markdown(
            "#### 🩺 Clinical Research Assessment"
        )

        st.metric(
            label=(
                "MACE Model Score "
                "(Positive Class = 1)"
            ),
            value=f"{model_score * 100:.4f}%",
        )

        st.info(
            "No independently validated clinical "
            "high-risk/low-risk threshold is currently "
            "available for this deployment. "
            "Therefore, the previous automatic 50% "
            "cutoff has been removed."
        )

        st.write(
            "A higher model score indicates a higher "
            "predicted model output for MACE. "
            "However, the score should not be "
            "interpreted as a calibrated absolute "
            "clinical event probability without "
            "additional validation."
        )

        st.caption(
            "Decision-tree models may return "
            "identical scores for small input "
            "changes that do not cross tree "
            "split thresholds."
        )

    with result_right:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=model_score * 100,
                number={
                    "suffix": "%",
                    "valueformat": ".2f",
                    "font": {
                        "size": 38,
                        "color": "#0A2540",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                    },
                    "bar": {
                        "color": "#1750A1",
                    },
                    "bgcolor": "#EFF5FB",
                    "borderwidth": 1,
                    "bordercolor": "#C9D6E4",
                },
            )
        )

        gauge.update_layout(
            height=275,
            margin={
                "l": 15,
                "r": 15,
                "t": 28,
                "b": 15,
            },
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )


    # ========================================================
    # 21. PREDICTION AUDIT
    # ========================================================

    with st.expander(
        "🔎 Prediction Audit: "
        "Raw Values and Model Inputs"
    ):

        audit_df = pd.DataFrame(
            {
                "Biomarker": FEATURES,

                "Raw Input": (
                    raw_input.iloc[0].to_numpy(
                        dtype=float
                    )
                ),

                "Training Mean": TRAIN_MEAN,

                "Training Scale": TRAIN_SCALE,

                "Actual Standardized Input": (
                    standardized_input.iloc[0].to_numpy(
                        dtype=float
                    )
                ),
            }
        )

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Standardized input = "
            "(Original value − training mean) "
            "/ training scale."
        )


    # ========================================================
    # 22. SHAP EXPLAINABILITY
    # ========================================================

    st.markdown(
        "### 🔍 Step 3: "
        "AI Explainability (SHAP Analysis)"
    )

    st.write(
        "The following visualization illustrates "
        "how each individual biomarker contributes "
        "to the model output relative to a "
        "reference prediction."
    )

    try:

        with st.spinner(
            "Calculating personalized "
            "SHAP explanations..."
        ):

            (
                shap_explanation,
                shap_units,
                reference_name,
            ) = explain_patient(
                model,
                standardized_input,
                raw_input,
                model_score,
            )

        st.caption(
            f"SHAP units: **{shap_units}**. "
            f"Reference: {reference_name}. "
            "Log-odds contributions should not "
            "be interpreted as percentage-point "
            "changes in clinical risk."
        )

        plot_left, plot_right = st.columns(
            [1.15, 1],
            gap="large",
        )

        with plot_left:

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

            shap_fig = plt.gcf()

            shap_fig.set_size_inches(
                7.5,
                5,
            )

            st.pyplot(
                shap_fig,
                use_container_width=True,
            )

            plt.close(
                shap_fig
            )

        with plot_right:

            st.markdown(
                "**Feature Contribution Ranking**"
            )

            contributions = np.asarray(
                shap_explanation.values,
                dtype=float,
            )

            abs_contributions = np.abs(
                contributions
            )

            total_contribution = (
                abs_contributions.sum()
            )

            if total_contribution > 0:

                contribution_shares = (
                    abs_contributions
                    / total_contribution
                    * 100
                )

            else:

                contribution_shares = (
                    np.zeros_like(
                        abs_contributions
                    )
                )

            effects = []

            for contribution in contributions:

                if contribution > 0:

                    effects.append(
                        "↑ Higher Model Output"
                    )

                elif contribution < 0:

                    effects.append(
                        "↓ Lower Model Output"
                    )

                else:

                    effects.append(
                        "No Effect"
                    )

            contribution_df = pd.DataFrame(
                {
                    "Biomarker": FEATURES,

                    "Patient Value": (
                        raw_input.iloc[0].to_numpy(
                            dtype=float
                        )
                    ),

                    "Effect": effects,

                    "SHAP Value": contributions,

                    "Contribution (%)": (
                        contribution_shares
                    ),
                }
            )

            contribution_df = (
                contribution_df.sort_values(
                    "Contribution (%)",
                    ascending=False,
                )
            )

            styled_df = (
                contribution_df.style.format(
                    {
                        "Patient Value": "{:.2f}",
                        "SHAP Value": "{:+.4f}",
                        "Contribution (%)": "{:.1f}%",
                    }
                )
            )

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "SHAP contributions explain "
                "model behavior. They do not "
                "establish causal clinical effects."
            )

    except Exception as exc:

        st.warning(
            "The prediction was calculated "
            "successfully, but SHAP visualization "
            "could not be completed in the current "
            f"environment: {exc}"
        )


# ============================================================
# 23. FOOTER
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

<br>

<i>
Powered by Streamlit |
Developed for Clinical Research
and Precision Medicine.
</i>

</div>
""",
    unsafe_allow_html=True,
)
