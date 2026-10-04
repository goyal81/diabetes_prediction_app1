# ============================================================
# DIABETES PREDICTION + XAI DASHBOARD
# ============================================================
# Model:
# Extra Trees + XGBoost -> MLP Meta Learner
#
# XAI:
# 1. SHAP
# 2. LIME Tabular
# 3. Counterfactual Explanation
# 4. Partial Dependence Plot (PDP)
# 5. XAI Interpretation / Chat
#
# The final dashboard builds the model itself at runtime.
# No saved model (.pkl/.joblib) is required.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.inspection import PartialDependenceDisplay

from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction & XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

.big-result {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
}

.small-note {
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Diabetes Prediction & XAI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Stacking Ensemble: Extra Trees + XGBoost + Neural Network'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = "df_balanced2.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    return data


try:

    df = load_data()

except Exception as e:

    st.error("Dataset could not be loaded.")

    st.code(DATA_PATH)

    st.exception(e)

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

@st.cache_resource
def prepare_dataset(df):

    target = "diabetes"

    X = df.drop(
        columns=[target]
    ).copy()

    y = df[target].copy()

    # --------------------------------------------------------
    # Force target numeric
    # --------------------------------------------------------

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert all X columns to numeric
    # --------------------------------------------------------

    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # Replace missing values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    continuous_features = [

        "BMI",

        "glucose",

        "systolic_avg",

        "diastolic_avg",

        "Hemoglobin level"

    ]

    continuous_features = [

        c for c in continuous_features

        if c in X_train.columns
    ]

    scaler = StandardScaler()

    X_train = X_train.copy()

    X_test = X_test.copy()

    if len(continuous_features) > 0:

        X_train[
            continuous_features
        ] = scaler.fit_transform(
            X_train[
                continuous_features
            ]
        )

        X_test[
            continuous_features
        ] = scaler.transform(
            X_test[
                continuous_features
            ]
        )

    # --------------------------------------------------------
    # FINAL NUMERIC CONVERSION
    # --------------------------------------------------------

    for col in X_train.columns:

        X_train[col] = pd.to_numeric(
            X_train[col],
            errors="coerce"
        )

        X_test[col] = pd.to_numeric(
            X_test[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Fill any remaining missing values
    # --------------------------------------------------------

    train_medians = X_train.median()

    X_train = X_train.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    # --------------------------------------------------------
    # Force float64
    # --------------------------------------------------------

    X_train = X_train.astype(
        np.float64
    )

    X_test = X_test.astype(
        np.float64
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        continuous_features
    )


(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    continuous_features
) = prepare_dataset(df)


feature_columns = list(
    X_train.columns
)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_stacking_model(
    X_train,
    y_train
):

    # --------------------------------------------------------
    # Extra Trees
    # --------------------------------------------------------

    et = ExtraTreesClassifier(
        random_state=42
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    best_xgb = XGBClassifier(

        n_estimators=797,

        max_depth=15,

        learning_rate=0.14683683239332332,

        subsample=0.8989586275610673,

        colsample_bytree=0.9198201644183858,

        gamma=0.08013822456836656,

        min_child_weight=9,

        reg_alpha=0.39446891483018054,

        reg_lambda=8.894450387968137,

        eval_metric="logloss",

        random_state=42,

        n_jobs=2
    )

    # --------------------------------------------------------
    # MLP Meta Learner
    # --------------------------------------------------------

    meta_nn = MLPClassifier(

        hidden_layer_sizes=(
            64,
            32,
            16
        ),

        activation="relu",

        solver="adam",

        alpha=0.001,

        batch_size=64,

        learning_rate_init=0.001,

        max_iter=200,

        random_state=42
    )

    # --------------------------------------------------------
    # Stacking
    # --------------------------------------------------------

    estimators = [

        (
            "et",
            et
        ),

        (
            "xgb",
            best_xgb
        )

    ]

    stack_nn = StackingClassifier(

        estimators=estimators,

        final_estimator=meta_nn,

        stack_method="predict_proba",

        passthrough=False,

        n_jobs=1
    )

    stack_nn.fit(
        X_train,
        y_train
    )

    return stack_nn


with st.spinner(
    "Training the stacking model from df_balanced2.csv... This may take a little while on first launch."
):

    stack_nn = train_stacking_model(
        X_train,
        y_train
    )


# ============================================================
# PERFORMANCE
# ============================================================

@st.cache_data
def calculate_metrics(
    _model,
    X_train,
    y_train,
    X_test,
    y_test
):

    train_pred = _model.predict(
        X_train
    )

    train_prob = _model.predict_proba(
        X_train
    )[:, 1]

    test_pred = _model.predict(
        X_test
    )

    test_prob = _model.predict_proba(
        X_test
    )[:, 1]

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report
    )

    metrics = {

        "Training Accuracy":
        accuracy_score(
            y_train,
            train_pred
        ),

        "Training Precision":
        precision_score(
            y_train,
            train_pred,
            zero_division=0
        ),

        "Training Recall":
        recall_score(
            y_train,
            train_pred,
            zero_division=0
        ),

        "Training F1":
        f1_score(
            y_train,
            train_pred,
            zero_division=0
        ),

        "Training ROC-AUC":
        roc_auc_score(
            y_train,
            train_prob
        ),

        "Testing Accuracy":
        accuracy_score(
            y_test,
            test_pred
        ),

        "Testing Precision":
        precision_score(
            y_test,
            test_pred,
            zero_division=0
        ),

        "Testing Recall":
        recall_score(
            y_test,
            test_pred,
            zero_division=0
        ),

        "Testing F1":
        f1_score(
            y_test,
            test_pred,
            zero_division=0
        ),

        "Testing ROC-AUC":
        roc_auc_score(
            y_test,
            test_prob
        )
    }

    cm = confusion_matrix(
        y_test,
        test_pred
    )

    report = classification_report(
        y_test,
        test_pred,
        output_dict=True
    )

    return (
        metrics,
        test_pred,
        test_prob,
        cm,
        report
    )


(
    metrics,
    test_predictions,
    test_probabilities,
    confusion,
    class_report
) = calculate_metrics(
    stack_nn,
    X_train,
    y_train,
    X_test,
    y_test
)


# ============================================================
# LIME EXPLAINER
# ============================================================

@st.cache_resource
def build_lime_explainer(
    X_train,
    feature_columns
):

    return LimeTabularExplainer(

        training_data=X_train.values,

        feature_names=feature_columns,

        class_names=[
            "No Diabetes",
            "Diabetes"
        ],

        mode="classification",

        discretize_continuous=True
    )


lime_explainer = build_lime_explainer(
    X_train,
    feature_columns
)


# ============================================================

# ============================================================
# SINGLE-PAGE EXPLAINABLE HEALTHCARE REPORT
# ============================================================

# Session state is used so the complete report is generated only after
# the user explicitly requests it and is retained across Streamlit reruns.
if "report_input_df" not in st.session_state:
    st.session_state.report_input_df = None
if "report_prediction" not in st.session_state:
    st.session_state.report_prediction = None
if "report_probabilities" not in st.session_state:
    st.session_state.report_probabilities = None
if "complete_xai_report" not in st.session_state:
    st.session_state.complete_xai_report = None

# ------------------------------------------------------------
# Extra report styling (does not change the model)
# ------------------------------------------------------------
st.markdown("""
<style>
.report-hero {
    padding: 28px 30px;
    border-radius: 20px;
    border: 1px solid rgba(100,100,100,.18);
    margin-bottom: 20px;
}
.report-hero h1 { margin: 0 0 8px 0; font-size: 34px; }
.report-hero p { margin: 0; font-size: 15px; opacity: .82; }
.result-card {
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(100,100,100,.18);
    text-align: center;
    min-height: 110px;
}
.result-label { font-size: 13px; opacity: .72; margin-bottom: 5px; }
.result-value { font-size: 27px; font-weight: 800; }
.report-note {
    padding: 14px 17px;
    border-left: 4px solid #5b7cfa;
    border-radius: 8px;
    background: rgba(91,124,250,.07);
    margin: 10px 0 18px 0;
}
.section-head {
    font-size: 24px;
    font-weight: 750;
    margin-top: 12px;
    margin-bottom: 4px;
}
.small-muted { opacity: .72; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="report-hero">
    <h1>🩺 Explainable Diabetes Prediction Report</h1>
    <p>Patient-level prediction with SHAP, LIME, counterfactual and partial-dependence explanations.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="report-note"><b>Research / decision-support use:</b> '
    'This report explains the behavior of the trained machine-learning model for the entered sample. '
    'It is not a medical diagnosis and the counterfactual results are not treatment recommendations.</div>',
    unsafe_allow_html=True
)

# ============================================================
# HELPER FUNCTIONS — XAI ONLY; MODEL IS UNCHANGED
# ============================================================

def _predict_positive(dataframe):
    """Positive-class probability for SHAP/LIME/PDP."""
    x = np.asarray(dataframe, dtype=np.float64)
    x = pd.DataFrame(x, columns=feature_columns).astype(np.float64)
    return stack_nn.predict_proba(x)[:, 1]


def _normalise_shap_values(values, n_rows, n_features):
    """Handle SHAP output differences across SHAP versions."""
    arr = np.asarray(values, dtype=np.float64)
    if isinstance(values, list):
        arr = np.asarray(values[-1], dtype=np.float64)
    if arr.ndim == 3:
        # Common forms: (classes, rows, features) or (rows, features, classes)
        if arr.shape[0] == 2 and arr.shape[1] == n_rows:
            arr = arr[1]
        elif arr.shape[-1] == 2 and arr.shape[0] == n_rows:
            arr = arr[:, :, 1]
        else:
            arr = arr.reshape(n_rows, n_features)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2 and arr.shape != (n_rows, n_features):
        arr = arr.reshape(n_rows, n_features)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _make_shap_explainer():
    background = X_train.sample(
        min(35, len(X_train)), random_state=42
    ).values.astype(np.float64)
    return shap.KernelExplainer(_predict_positive, background)


def _raw_value(feature, value):
    """Convert a scaled continuous feature back to its original value for display."""
    try:
        if feature in continuous_features:
            idx = continuous_features.index(feature)
            mean = scaler.mean_[idx]
            scale = scaler.scale_[idx]
            return float(value * scale + mean)
    except Exception:
        pass
    return float(value)


def _pdp_values(model, data, feature, grid_points=15):
    """Manual PDP calculation; avoids estimator/version-specific PDP issues."""
    base = data.copy().astype(np.float64)
    col = base[feature]
    unique = np.sort(col.dropna().unique())
    if len(unique) <= 2 and len(unique) <= grid_points:
        grid = unique
    else:
        grid = np.unique(np.quantile(col, np.linspace(0.02, 0.98, grid_points)))
    values = []
    for v in grid:
        temp = base.copy()
        temp[feature] = v
        values.append(float(model.predict_proba(temp)[:, 1].mean()))
    return grid, np.asarray(values)


def _counterfactual_search(original, original_prediction, allowed_features,
                           max_change=1.5, step=0.10):
    """Find the first one-feature model-space change that flips the prediction."""
    best = None
    for feature in allowed_features:
        old = float(original[feature])
        # Avoid nonsensical search on binary one-hot variables.
        if set(np.unique(X_train[feature].values)).issubset({0.0, 1.0}):
            candidates = [0.0, 1.0]
        else:
            candidates = np.arange(old - max_change, old + max_change + step/2, step)
        for new_value in candidates:
            if np.isclose(new_value, old):
                continue
            candidate = original.copy()
            candidate[feature] = float(new_value)
            candidate_df = pd.DataFrame([candidate], columns=feature_columns).astype(np.float64)
            pred = int(stack_nn.predict(candidate_df)[0])
            if pred != original_prediction:
                old_prob = float(stack_nn.predict_proba(pd.DataFrame([original], columns=feature_columns))[0, 1])
                new_prob = float(stack_nn.predict_proba(candidate_df)[0, 1])
                distance = abs(float(new_value) - old)
                candidate_result = (distance, feature, old, float(new_value), pred, new_prob, candidate_df)
                if best is None or distance < best[0]:
                    best = candidate_result
                break
    return best


# ============================================================
# INPUT / PREDICTION SECTION
# ============================================================
st.markdown('<div class="section-head">1. Patient / Sample Input</div>', unsafe_allow_html=True)
st.caption("Enter the same variables used by the original model. The model architecture and preprocessing are unchanged.")

input_data = {}
numeric_features = [
    "age", "education_level", "BMI", "smoking", "alcohol",
    "hypertension", "heart_disease", "bp_cuff_size", "glucose",
    "wealth_index", "systolic_avg", "diastolic_avg",
    "Hemoglobin level", "Anemia level"
]
available_numeric = [f for f in numeric_features if f in feature_columns]

cols = st.columns(3)
for i, feature in enumerate(available_numeric):
    with cols[i % 3]:
        default = float(X_train[feature].median())
        input_data[feature] = st.number_input(feature, value=default, format="%.4f", key=f"report_{feature}")

# Categorical variables are retained exactly in the spirit of the original UI.
if "residence_type_Urban" in feature_columns or "residence_type_Rural" in feature_columns:
    residence = st.selectbox("Residence", ["Urban", "Rural"], key="report_residence")
    if "residence_type_Urban" in feature_columns:
        input_data["residence_type_Urban"] = int(residence == "Urban")
    if "residence_type_Rural" in feature_columns:
        input_data["residence_type_Rural"] = int(residence == "Rural")

if "gender_Female" in feature_columns or "gender_Male" in feature_columns or "gender_Other" in feature_columns:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="report_gender")
    for col in ["gender_Female", "gender_Male", "gender_Other"]:
        if col in feature_columns:
            input_data[col] = int(col == f"gender_{gender}")

marital_columns = [
    "marital_status_Divorced/Separated", "marital_status_Married",
    "marital_status_Never married", "marital_status_Widowed"
]
if any(x in feature_columns for x in marital_columns):
    marital = st.selectbox(
        "Marital Status",
        ["Married", "Never married", "Widowed", "Divorced/Separated"],
        key="report_marital"
    )
    for col in marital_columns:
        if col in feature_columns:
            input_data[col] = int(col.replace("marital_status_", "") == marital)

for feature in feature_columns:
    if feature not in input_data:
        input_data[feature] = float(X_train[feature].median())

raw_input = pd.DataFrame([input_data])[feature_columns]
for col in raw_input.columns:
    raw_input[col] = pd.to_numeric(raw_input[col], errors="coerce")
raw_input = raw_input.fillna(X_train.median())
input_scaled = raw_input.copy()
if len(continuous_features) > 0:
    input_scaled[continuous_features] = scaler.transform(input_scaled[continuous_features])
input_scaled = input_scaled.astype(np.float64)

if st.button("🚀 Generate Complete Prediction + XAI Report", type="primary", use_container_width=True):
    with st.spinner("Generating prediction and complete explainable report..."):
        pred = int(stack_nn.predict(input_scaled)[0])
        probs = stack_nn.predict_proba(input_scaled)[0]
        st.session_state.report_input_df = input_scaled.copy()
        st.session_state.report_prediction = pred
        st.session_state.report_probabilities = probs
        st.session_state.complete_xai_report = None

        # --------------------------------------------------------
        # SHAP local + small global sample
        # --------------------------------------------------------
        explainer = _make_shap_explainer()
        local_raw = explainer.shap_values(input_scaled.values.astype(np.float64), nsamples=150)
        local_sv = _normalise_shap_values(local_raw, 1, len(feature_columns))[0]
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        local_exp = shap.Explanation(
            values=local_sv,
            base_values=expected,
            data=input_scaled.iloc[0].values,
            feature_names=feature_columns
        )

        global_test = X_test.sample(min(18, len(X_test)), random_state=42)
        global_raw = explainer.shap_values(global_test.values.astype(np.float64), nsamples=60)
        global_sv = _normalise_shap_values(global_raw, len(global_test), len(feature_columns))

        shap_df = pd.DataFrame({
            "Feature": feature_columns,
            "Model value": input_scaled.iloc[0].values,
            "SHAP value": local_sv,
            "Absolute SHAP": np.abs(local_sv)
        }).sort_values("Absolute SHAP", ascending=False)

        # --------------------------------------------------------
        # LIME
        # --------------------------------------------------------
        lime_exp = lime_explainer.explain_instance(
            input_scaled.iloc[0].values.astype(np.float64),
            stack_nn.predict_proba,
            num_features=min(10, len(feature_columns)),
            num_samples=3000
        )
        lime_pairs = lime_exp.as_list(label=1)
        lime_df = pd.DataFrame(lime_pairs, columns=["Feature / Condition", "Weight"])
        lime_df["Direction"] = np.where(
            lime_df["Weight"] > 0,
            "Supports Diabetes",
            "Supports No Diabetes"
        )
        lime_df = lime_df.sort_values("Weight", ascending=True)

        # --------------------------------------------------------
        # Counterfactual
        # --------------------------------------------------------
        cf_features = [
            f for f in ["glucose", "BMI", "systolic_avg", "diastolic_avg",
                        "Hemoglobin level", "age", "hypertension", "heart_disease"]
            if f in feature_columns
        ]
        cf = _counterfactual_search(
            input_scaled.iloc[0].copy(), pred, cf_features,
            max_change=1.5, step=0.10
        )

        # --------------------------------------------------------
        # PDP for the most useful available features
        # --------------------------------------------------------
        preferred_pdp = [f for f in ["glucose", "BMI", "age", "systolic_avg", "diastolic_avg"] if f in feature_columns]
        pdp_features = preferred_pdp[:3]

        st.session_state.complete_xai_report = {
            "explainer": explainer,
            "local_exp": local_exp,
            "local_sv": local_sv,
            "global_test": global_test,
            "global_sv": global_sv,
            "shap_df": shap_df,
            "lime_exp": lime_exp,
            "lime_df": lime_df,
            "counterfactual": cf,
            "pdp_features": pdp_features
        }

# ============================================================
# COMPLETE REPORT
# ============================================================
if st.session_state.report_input_df is not None and st.session_state.complete_xai_report is not None:

    report_input = st.session_state.report_input_df
    prediction = int(st.session_state.report_prediction)
    probabilities = np.asarray(st.session_state.report_probabilities)
    report = st.session_state.complete_xai_report

    st.divider()
    st.markdown('<div class="section-head">2. Prediction Summary</div>', unsafe_allow_html=True)

    prediction_label = "Diabetes" if prediction == 1 else "No Diabetes"
    confidence = float(np.max(probabilities))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="result-card"><div class="result-label">MODEL PREDICTION</div><div class="result-value">{prediction_label}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="result-card"><div class="result-label">DIABETES PROBABILITY</div><div class="result-value">{probabilities[1]*100:.2f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="result-card"><div class="result-label">MODEL CONFIDENCE</div><div class="result-value">{confidence*100:.2f}%</div></div>', unsafe_allow_html=True)

    st.markdown("### Prediction probability")
    prob_df = pd.DataFrame({
        "Class": ["No Diabetes", "Diabetes"],
        "Probability": probabilities * 100
    })
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(prob_df["Class"], prob_df["Probability"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model class probabilities")
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, prob_df["Probability"]):
        ax.text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ========================================================
    # XAI INTERPRETATION SUMMARY
    # ========================================================
    st.markdown('<div class="section-head">3. XAI Interpretation Summary</div>', unsafe_allow_html=True)

    top_shap = report["shap_df"].head(5)
    top_lime_positive = report["lime_df"].sort_values("Weight", ascending=False).head(3)
    top_lime_negative = report["lime_df"].sort_values("Weight", ascending=True).head(3)

    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown("#### SHAP — strongest local influences")
        for _, row in top_shap.iterrows():
            direction = "toward Diabetes" if row["SHAP value"] > 0 else "toward No Diabetes"
            st.write(f"**{row['Feature']}** → {direction} ({row['SHAP value']:.4f})")
    with summary_cols[1]:
        st.markdown("#### LIME — strongest local rules")
        for _, row in pd.concat([top_lime_positive, top_lime_negative]).drop_duplicates().iterrows():
            direction = "toward Diabetes" if row["Weight"] > 0 else "toward No Diabetes"
            st.write(f"**{row['Feature / Condition']}** → {direction} ({row['Weight']:.4f})")

    st.info(
        "Interpretation: SHAP and LIME describe how the model used the selected input locally. "
        "Agreement between methods can strengthen understanding of model behavior, but neither method establishes causation."
    )

    # ========================================================
    # SHAP
    # ========================================================
    st.markdown('<div class="section-head">4. SHAP Explanation</div>', unsafe_allow_html=True)
    shap_tabs = st.tabs(["Waterfall", "Local contribution", "Global bar", "Beeswarm", "Dependence", "Table"])

    with shap_tabs[0]:
        fig = plt.figure(figsize=(11, 6.5))
        shap.plots.waterfall(report["local_exp"], max_display=min(15, len(feature_columns)), show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with shap_tabs[1]:
        plot_df = report["shap_df"].head(min(15, len(feature_columns))).sort_values("SHAP value")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["tab:orange" if x > 0 else "tab:blue" for x in plot_df["SHAP value"]]
        ax.barh(plot_df["Feature"], plot_df["SHAP value"], color=colors)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("SHAP value")
        ax.set_title("Local SHAP contributions")
        ax.grid(axis="x", alpha=.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with shap_tabs[2]:
        mean_abs = np.abs(report["global_sv"]).mean(axis=0)
        global_df = pd.DataFrame({"Feature": feature_columns, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(global_df["Feature"], global_df["Mean |SHAP|"])
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("Global feature importance on the selected test sample")
        ax.grid(axis="x", alpha=.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with shap_tabs[3]:
        global_exp = shap.Explanation(
            values=report["global_sv"],
            data=report["global_test"].values,
            feature_names=feature_columns
        )
        fig = plt.figure(figsize=(11, 7))
        shap.plots.beeswarm(global_exp, max_display=min(15, len(feature_columns)), show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with shap_tabs[4]:
        selected_shap_feature = st.selectbox("SHAP dependence feature", feature_columns, key="report_shap_dep")
        idx = feature_columns.index(selected_shap_feature)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(report["global_test"][selected_shap_feature], report["global_sv"][:, idx], alpha=.75)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel(selected_shap_feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence: {selected_shap_feature}")
        ax.grid(alpha=.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with shap_tabs[5]:
        st.dataframe(report["shap_df"].round(5), use_container_width=True, hide_index=True)

    # ========================================================
    # LIME
    # ========================================================
    st.markdown('<div class="section-head">5. LIME Explanation</div>', unsafe_allow_html=True)
    lime_tabs = st.tabs(["LIME plot", "Feature values", "Contribution chart", "Weights"])

    with lime_tabs[0]:
        fig = report["lime_exp"].as_pyplot_figure(label=1)
        fig.set_size_inches(11, 6.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("This is the standard LIME local explanation: feature conditions and their contribution to the selected class.")

    with lime_tabs[1]:
        feature_values = pd.DataFrame({
            "Feature": feature_columns,
            "Value (model input)": report_input.iloc[0].values,
            "Value (display scale)": [_raw_value(f, report_input.iloc[0][f]) for f in feature_columns]
        })
        st.dataframe(feature_values.round(4), use_container_width=True, hide_index=True)

    with lime_tabs[2]:
        ldf = report["lime_df"]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["tab:orange" if x > 0 else "tab:blue" for x in ldf["Weight"]]
        ax.barh(ldf["Feature / Condition"], ldf["Weight"], color=colors)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("LIME weight")
        ax.set_title("Local LIME feature contributions")
        ax.grid(axis="x", alpha=.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with lime_tabs[3]:
        st.dataframe(report["lime_df"].round(5), use_container_width=True, hide_index=True)

    # ========================================================
    # COUNTERFACTUAL
    # ========================================================
    st.markdown('<div class="section-head">6. Counterfactual Explanation</div>', unsafe_allow_html=True)
    cf = report["counterfactual"]
    if cf is None:
        st.warning("No one-feature counterfactual was found within the predefined model-space search range.")
    else:
        _, feature, old_value, new_value, new_prediction, new_probability, candidate_df = cf
        old_probability = float(probabilities[1])
        st.success(
            f"A model-space change in **{feature}** from **{old_value:.4f}** to **{new_value:.4f}** "
            f"flips the model prediction from **{'Diabetes' if prediction == 1 else 'No Diabetes'}** "
            f"to **{'Diabetes' if new_prediction == 1 else 'No Diabetes'}**."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Feature changed", feature)
        with c2:
            st.metric("Original probability", f"{old_probability*100:.2f}%")
        with c3:
            st.metric("Counterfactual probability", f"{new_probability*100:.2f}%")

        cf_compare = pd.DataFrame({
            "Feature": [feature],
            "Original (model input)": [old_value],
            "Counterfactual (model input)": [new_value],
            "Original (display scale)": [_raw_value(feature, old_value)],
            "Counterfactual (display scale)": [_raw_value(feature, new_value)]
        })
        st.dataframe(cf_compare.round(4), use_container_width=True, hide_index=True)

        prob_compare = pd.DataFrame({
            "Scenario": ["Original", "Counterfactual"],
            "Diabetes probability": [old_probability*100, new_probability*100]
        })
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(prob_compare["Scenario"], prob_compare["Diabetes probability"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Diabetes probability (%)")
        ax.set_title("Original vs counterfactual model response")
        for bar, value in zip(bars, prob_compare["Diabetes probability"]):
            ax.text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.1f}%", ha="center", fontweight="bold")
        ax.grid(axis="y", alpha=.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption("Counterfactuals describe a hypothetical model response. They are not recommendations to change a person's health measurements or treatment.")

    # ========================================================
    # PDP
    # ========================================================
    st.markdown('<div class="section-head">7. Partial Dependence Analysis</div>', unsafe_allow_html=True)
    st.caption("PDP shows the model's average predicted diabetes probability as one feature varies while the other test-set features are retained.")
    pdp_grid = st.slider("PDP grid points", 8, 20, 12, key="report_pdp_grid")
    for feature in report["pdp_features"]:
        grid, pdp_prob = _pdp_values(stack_nn, X_test, feature, grid_points=pdp_grid)
        fig, ax = plt.subplots(figsize=(9, 4.7))
        ax.plot(grid, pdp_prob * 100, marker="o")
        ax.set_xlabel(feature)
        ax.set_ylabel("Average diabetes probability (%)")
        ax.set_title(f"Partial Dependence: {feature}")
        ax.grid(alpha=.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ========================================================
    # FINAL CLINICAL-STYLE XAI REPORT
    # ========================================================
    st.markdown('<div class="section-head">8. Integrated XAI Report</div>', unsafe_allow_html=True)

    top_shap_row = report["shap_df"].iloc[0]
    strongest_direction = "increased the model's diabetes prediction" if top_shap_row["SHAP value"] > 0 else "reduced the model's diabetes prediction"
    cf_sentence = "A counterfactual was identified within the search range." if cf is not None else "No counterfactual was identified within the predefined search range."

    st.markdown(f"""
    <div class="report-note">
    <b>Prediction:</b> The stacking model predicts <b>{prediction_label}</b> with a diabetes probability of <b>{probabilities[1]*100:.2f}%</b>.<br><br>
    <b>SHAP:</b> The strongest local feature was <b>{top_shap_row['Feature']}</b>, which {strongest_direction} (SHAP = {top_shap_row['SHAP value']:.4f}).<br><br>
    <b>LIME:</b> The local surrogate explanation identifies feature conditions that support or oppose the predicted class around this individual sample.<br><br>
    <b>Counterfactual:</b> {cf_sentence}<br><br>
    <b>PDP:</b> The partial-dependence plots show how the trained model's average probability responds to changes in selected features; they do not establish causal relationships.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model performance context")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Test Accuracy", f"{metrics['Testing Accuracy']*100:.2f}%")
    with mc2:
        st.metric("Test F1 Score", f"{metrics['Testing F1']*100:.2f}%")
    with mc3:
        st.metric("Test ROC-AUC", f"{metrics['Testing ROC-AUC']:.3f}")

    st.caption("Only Accuracy, F1 Score and ROC-AUC are shown as model-performance metrics in this report. Training performance, precision, recall, sensitivity and specificity are not displayed.")

    st.success("Complete explainable prediction report generated successfully.")
