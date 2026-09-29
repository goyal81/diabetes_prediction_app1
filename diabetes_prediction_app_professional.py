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
st.set_page_config(
    page_title="Explainable Diabetes Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ============================================================
# PROFESSIONAL RESEARCH DASHBOARD UI
# ============================================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Session state
# -----------------------------
for key, default in {
    "input_df": None,
    "raw_input_df": None,
    "prediction": None,
    "probabilities": None,
    "shap_result": None,
    "lime_result": None,
    "counterfactual": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------
# Professional styling
# -----------------------------
st.markdown("""
<style>
:root {
    --navy: #0b1f3a;
    --blue: #2563eb;
    --cyan: #0891b2;
    --green: #059669;
    --red: #dc2626;
    --slate: #475569;
    --border: #e2e8f0;
    --soft: #f8fafc;
}
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1500px;}
.main-title {font-size: 2.35rem; font-weight: 800; color: var(--navy); margin: 0;}
.subtitle {color: #64748b; font-size: 1.03rem; margin-top: .2rem; margin-bottom: 1.2rem;}
.section-title {font-size: 1.35rem; font-weight: 750; color: var(--navy); margin-top: .8rem;}
.hero {
    padding: 1.4rem 1.6rem; border-radius: 18px; border: 1px solid var(--border);
    background: linear-gradient(135deg,#f8fbff,#eef6ff);
    margin-bottom: 1rem;
}
.card {
    padding: 1rem 1.1rem; border-radius: 14px; border: 1px solid var(--border);
    background: white; min-height: 90px;
}
.kpi-label {font-size:.78rem; color:#64748b; text-transform:uppercase; letter-spacing:.06em;}
.kpi-value {font-size:1.55rem; font-weight:800; color:var(--navy); margin-top:.15rem;}
.kpi-note {font-size:.76rem; color:#64748b; margin-top:.15rem;}
.badge {
    display:inline-block; padding:.28rem .65rem; border-radius:999px;
    font-size:.75rem; font-weight:700; background:#e0f2fe; color:#075985;
}
.disclaimer {
    padding:.8rem 1rem; border-left:4px solid #64748b; background:#f8fafc;
    color:#475569; border-radius:8px; font-size:.84rem;
}
.sidebar-note {font-size:.78rem; color:#64748b;}
div[data-testid="stMetric"] {border:1px solid var(--border); padding:.7rem; border-radius:12px; background:white;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="main-title">🧠 Explainable Diabetes Prediction</div>
  <div class="subtitle">Research-oriented clinical decision-support interface · Stacking ensemble + multi-method XAI</div>
  <span class="badge">Extra Trees + XGBoost → MLP Meta Learner</span>
  <span class="badge">Test-set evaluation</span>
  <span class="badge">SHAP · LIME · Counterfactual · PDP</span>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Cached evaluation
# -----------------------------
@st.cache_data
def calculate_test_metrics(_model, X_test, y_test):
    pred = _model.predict(X_test)
    prob = _model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall / Sensitivity": recall_score(y_test, pred, zero_division=0),
        "Specificity": (cm[0,0] / cm[0].sum()) if cm[0].sum() else 0,
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "Brier Score": brier_score_loss(y_test, prob),
    }
    return metrics, pred, prob, cm, report

metrics, test_predictions, test_probabilities, confusion, class_report = calculate_test_metrics(
    stack_nn, X_test, y_test
)

@st.cache_data
def base_model_test_metrics(_model, X_test, y_test):
    et = _model.named_estimators_["et"]
    xgb = _model.named_estimators_["xgb"]
    rows = []
    for name, mdl in [("Extra Trees", et), ("XGBoost", xgb), ("Stacking Ensemble", _model)]:
        pred = mdl.predict(X_test)
        prob = mdl.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
    return pd.DataFrame(rows)

model_compare = base_model_test_metrics(stack_nn, X_test, y_test)

# -----------------------------
# Helper plotting functions
# -----------------------------
def clean_fig(fig):
    fig.patch.set_alpha(0)
    return fig

def metric_cards(metric_dict):
    cols = st.columns(len(metric_dict))
    for col, (label, value) in zip(cols, metric_dict.items()):
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-note">Independent test set</div>
            </div>
            """, unsafe_allow_html=True)

def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    im = ax.imshow(cm)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_yticklabels(["No Diabetes", "Diabetes"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    ax.set_title("Test-set confusion matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    return fig

def probability_histogram(y, prob):
    fig, ax = plt.subplots(figsize=(9,4.8))
    ax.hist(prob[y.values == 0], bins=15, alpha=.65, label="No Diabetes")
    ax.hist(prob[y.values == 1], bins=15, alpha=.65, label="Diabetes")
    ax.axvline(.5, linestyle="--", linewidth=1.5, label="0.50 threshold")
    ax.set_xlabel("Predicted probability of diabetes")
    ax.set_ylabel("Number of test observations")
    ax.set_title("Distribution of test-set predicted probabilities", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig

# -----------------------------
# Navigation
# -----------------------------
st.sidebar.markdown("## 🧭 Research Dashboard")
st.sidebar.caption("Professional XAI interface")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🔮 Patient Prediction",
        "📊 SHAP Analysis",
        "🍋 LIME Analysis",
        "🔄 Counterfactual",
        "📈 PDP Analysis",
        "📋 Model Performance",
        "🧩 Model Architecture",
        "ℹ️ Research Notes",
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-note"><b>Evaluation:</b> all reported performance metrics below refer to the held-out test set. Training-set performance is intentionally not displayed.</div>',
    unsafe_allow_html=True
)

# ============================================================
# OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown("## Research dashboard overview")
    st.write("A consolidated view of prediction performance, model behavior and explainability outputs.")

    metric_cards({
        "Test accuracy": f"{metrics['Accuracy']*100:.2f}%",
        "Test F1": f"{metrics['F1 Score']*100:.2f}%",
        "Test ROC-AUC": f"{metrics['ROC-AUC']:.3f}",
        "Sensitivity": f"{metrics['Recall / Sensitivity']*100:.2f}%",
        "Specificity": f"{metrics['Specificity']*100:.2f}%",
        "Features": str(len(feature_columns))
    })

    st.markdown("### Model evaluation at a glance")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_confusion(confusion), use_container_width=True)
    with c2:
        st.pyplot(probability_histogram(y_test, test_probabilities), use_container_width=True)

    st.markdown("### Test-set model comparison")
    compare_display = model_compare.copy()
    for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"]:
        compare_display[c] = compare_display[c].map(lambda x: f"{x:.3f}")
    st.dataframe(compare_display, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer"><b>Interpretation:</b> model predictions and XAI explanations describe learned statistical behavior of the trained model. They are not a clinical diagnosis and should be interpreted alongside professional medical assessment.</div>
    """, unsafe_allow_html=True)

# ============================================================
# PATIENT PREDICTION
# ============================================================
elif page == "🔮 Patient Prediction":
    st.markdown("## Patient-level prediction")
    st.caption("Enter a sample profile, obtain the model probability, then use the XAI pages to investigate why the model produced that result.")

    input_data = {}
    numeric_features = [
        "age","education_level","BMI","smoking","alcohol","hypertension",
        "heart_disease","bp_cuff_size","glucose","wealth_index",
        "systolic_avg","diastolic_avg","Hemoglobin level","Anemia level"
    ]
    available_numeric = [f for f in numeric_features if f in feature_columns]

    with st.expander("Patient profile", expanded=True):
        cols = st.columns(3)
        for i, feature in enumerate(available_numeric):
            with cols[i % 3]:
                default = float(X_train[feature].median())
                lo = float(X_train[feature].min())
                hi = float(X_train[feature].max())
                input_data[feature] = st.number_input(
                    feature, value=default, min_value=lo, max_value=hi,
                    format="%.4f"
                )

        if "residence_type_Urban" in feature_columns or "residence_type_Rural" in feature_columns:
            residence = st.selectbox("Residence", ["Urban","Rural"])
            if "residence_type_Urban" in feature_columns:
                input_data["residence_type_Urban"] = int(residence == "Urban")
            if "residence_type_Rural" in feature_columns:
                input_data["residence_type_Rural"] = int(residence == "Rural")

        if any(x in feature_columns for x in ["gender_Female","gender_Male","gender_Other"]):
            gender = st.selectbox("Gender", ["Female","Male","Other"])
            for col in ["gender_Female","gender_Male","gender_Other"]:
                if col in feature_columns:
                    input_data[col] = int(gender == col.replace("gender_",""))

        marital_columns = [
            "marital_status_Divorced/Separated","marital_status_Married",
            "marital_status_Never married","marital_status_Widowed"
        ]
        if any(x in feature_columns for x in marital_columns):
            marital = st.selectbox("Marital status", ["Married","Never married","Widowed","Divorced/Separated"])
            for col in marital_columns:
                if col in feature_columns:
                    input_data[col] = int(col.replace("marital_status_","") == marital)

    for feature in feature_columns:
        if feature not in input_data:
            input_data[feature] = float(X_train[feature].median())

    raw_input = pd.DataFrame([input_data])[feature_columns]
    raw_input = raw_input.apply(pd.to_numeric, errors="coerce").fillna(X_train.median())
    input_scaled = raw_input.copy()
    if continuous_features:
        input_scaled[continuous_features] = scaler.transform(input_scaled[continuous_features])
    input_scaled = input_scaled.astype(np.float64)

    if st.button("🚀 Run diabetes prediction", type="primary", use_container_width=True):
        prediction = int(stack_nn.predict(input_scaled)[0])
        probabilities = stack_nn.predict_proba(input_scaled)[0]
        st.session_state.raw_input_df = raw_input
        st.session_state.input_df = input_scaled
        st.session_state.prediction = prediction
        st.session_state.probabilities = probabilities
        st.session_state.shap_result = None
        st.session_state.lime_result = None
        st.session_state.counterfactual = None

    if st.session_state.prediction is not None:
        prediction = st.session_state.prediction
        probabilities = st.session_state.probabilities

        st.markdown("### Prediction result")
        metric_cards({
            "Predicted class": "Diabetes" if prediction == 1 else "No Diabetes",
            "Diabetes probability": f"{probabilities[1]*100:.2f}%",
            "No-diabetes probability": f"{probabilities[0]*100:.2f}%",
            "Model confidence": f"{max(probabilities)*100:.2f}%"
        })

        if prediction == 1:
            st.error("Model output: Diabetes (Class 1)")
        else:
            st.success("Model output: No Diabetes (Class 0)")

        prob_df = pd.DataFrame({
            "Class":["No Diabetes","Diabetes"],
            "Probability":[probabilities[0],probabilities[1]]
        })
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.bar(prob_df["Class"], prob_df["Probability"])
        ax.set_ylim(0,1)
        ax.set_ylabel("Predicted probability")
        ax.set_title("Patient-level class probabilities", fontweight="bold")
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(i, v+.025, f"{v*100:.1f}%", ha="center", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        with st.expander("View processed model input"):
            st.dataframe(input_scaled.T.rename(columns={0:"Value"}), use_container_width=True)

# ============================================================
# SHAP
# ============================================================
elif page == "📊 SHAP Analysis":
    st.markdown("## SHAP explainability")
    st.caption("SHAP quantifies how individual features contribute to the model's predicted diabetes probability. Positive values push toward Class 1; negative values push toward Class 0.")

    if st.session_state.input_df is None:
        st.info("Run a prediction first.")
    else:
        shap_view = st.selectbox(
            "Visualization",
            ["Local contribution", "Waterfall", "Global bar", "Beeswarm", "Dependence", "Heatmap"]
        )

        @st.cache_resource
        def make_kernel_explainer(_X_train, _model):
            background = _X_train.sample(min(40, len(_X_train)), random_state=42).values.astype(np.float64)
            def model_positive(X):
                X = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=feature_columns)
                return _model.predict_proba(X)[:,1]
            return shap.KernelExplainer(model_positive, background)

        if st.button("Generate SHAP analysis", type="primary"):
            with st.spinner("Computing SHAP values..."):
                explainer = make_kernel_explainer(X_train, stack_nn)
                sv = explainer.shap_values(
                    st.session_state.input_df.values.astype(np.float64),
                    nsamples=120
                )
                sv = np.asarray(sv, dtype=np.float64)
                if sv.ndim == 3:
                    sv = sv[0,:,0]
                elif sv.ndim == 2:
                    sv = sv[0]
                sv = np.nan_to_num(sv)
                expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
                explanation = shap.Explanation(
                    values=sv, base_values=expected,
                    data=st.session_state.input_df.iloc[0].values,
                    feature_names=feature_columns
                )
                shap_df = pd.DataFrame({
                    "Feature":feature_columns,
                    "Processed value":st.session_state.input_df.iloc[0].values,
                    "SHAP value":sv,
                    "Absolute SHAP":np.abs(sv)
                }).sort_values("Absolute SHAP", ascending=False)
                st.session_state.shap_result = (explainer, explanation, shap_df)

        if st.session_state.shap_result is not None:
            explainer, explanation, shap_df = st.session_state.shap_result
            top = shap_df.head(min(15,len(shap_df)))

            st.markdown("### Local contribution profile")
            c1, c2 = st.columns([1.4,1])
            with c1:
                plot_df = top.sort_values("SHAP value")
                fig, ax = plt.subplots(figsize=(9,6.5))
                ax.barh(plot_df["Feature"], plot_df["SHAP value"])
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("SHAP value (effect on diabetes probability)")
                ax.set_title("Top local feature contributions", fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
            with c2:
                st.markdown("#### Evidence summary")
                pos = top[top["SHAP value"] > 0].head(5)
                neg = top[top["SHAP value"] < 0].head(5)
                st.write("**Pushes toward Diabetes**")
                for _, r in pos.iterrows():
                    st.write(f"• {r['Feature']}: +{r['SHAP value']:.4f}")
                st.write("**Pushes toward No Diabetes**")
                for _, r in neg.iterrows():
                    st.write(f"• {r['Feature']}: {r['SHAP value']:.4f}")

            if shap_view == "Waterfall":
                fig = plt.figure(figsize=(10,7))
                shap.plots.waterfall(explanation, max_display=15, show=False)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True)

            elif shap_view == "Global bar":
                # Approximate global SHAP importance using a controlled test sample.
                sample = X_test.sample(min(30,len(X_test)), random_state=42)
                multi = np.asarray(explainer.shap_values(sample.values.astype(np.float64), nsamples=60))
                if multi.ndim == 3: multi = multi[:,:,0]
                if multi.ndim == 1: multi = multi.reshape(1,-1)
                imp = pd.DataFrame({"Feature":feature_columns,"Mean |SHAP|":np.abs(multi).mean(axis=0)})
                imp = imp.sort_values("Mean |SHAP|", ascending=True).tail(15)
                fig, ax = plt.subplots(figsize=(9,6))
                ax.barh(imp["Feature"], imp["Mean |SHAP|"])
                ax.set_xlabel("Mean absolute SHAP value")
                ax.set_title("Global SHAP importance on test sample", fontweight="bold")
                fig.tight_layout(); st.pyplot(fig, use_container_width=True)

            elif shap_view == "Beeswarm":
                sample = X_test.sample(min(25,len(X_test)), random_state=42)
                multi = np.asarray(explainer.shap_values(sample.values.astype(np.float64), nsamples=50))
                if multi.ndim == 3: multi = multi[:,:,0]
                if multi.ndim == 1: multi = multi.reshape(1,-1)
                exp = shap.Explanation(values=multi, data=sample.values, feature_names=feature_columns)
                plt.figure(figsize=(10,7))
                shap.plots.beeswarm(exp, max_display=15, show=False)
                plt.tight_layout(); st.pyplot(plt.gcf(), use_container_width=True)

            elif shap_view == "Dependence":
                feature = st.selectbox("Feature for SHAP dependence", feature_columns)
                idx = feature_columns.index(feature)
                sample = X_test.sample(min(30,len(X_test)), random_state=42)
                multi = np.asarray(explainer.shap_values(sample.values.astype(np.float64), nsamples=50))
                if multi.ndim == 3: multi = multi[:,:,0]
                if multi.ndim == 1: multi = multi.reshape(1,-1)
                plt.figure(figsize=(9,6))
                shap.dependence_plot(idx, multi, sample, feature_names=feature_columns, show=False)
                plt.tight_layout(); st.pyplot(plt.gcf(), use_container_width=True)

            elif shap_view == "Heatmap":
                sample = X_test.sample(min(25,len(X_test)), random_state=42)
                multi = np.asarray(explainer.shap_values(sample.values.astype(np.float64), nsamples=50))
                if multi.ndim == 3: multi = multi[:,:,0]
                if multi.ndim == 1: multi = multi.reshape(1,-1)
                exp = shap.Explanation(values=multi, data=sample.values, feature_names=feature_columns)
                plt.figure(figsize=(12,7))
                shap.plots.heatmap(exp, max_display=15, show=False)
                plt.tight_layout(); st.pyplot(plt.gcf(), use_container_width=True)

            st.markdown("### SHAP contribution table")
            st.dataframe(top, use_container_width=True, hide_index=True)

# ============================================================
# LIME
# ============================================================
elif page == "🍋 LIME Analysis":
    st.markdown("## LIME local explanation")
    st.caption("LIME fits a local surrogate around the selected patient/sample. The weights are local model-approximation coefficients, not causal effects.")

    if st.session_state.input_df is None:
        st.info("Run a prediction first.")
    else:
        if st.button("Generate LIME explanation", type="primary"):
            with st.spinner("Generating local LIME explanation..."):
                lime_explainer = LimeTabularExplainer(
                    training_data=X_train.values,
                    feature_names=feature_columns,
                    class_names=["No Diabetes","Diabetes"],
                    mode="classification",
                    discretize_continuous=True,
                    random_state=42
                )
                exp = lime_explainer.explain_instance(
                    st.session_state.input_df.iloc[0].values.astype(np.float64),
                    stack_nn.predict_proba,
                    num_features=min(12,len(feature_columns)),
                    num_samples=2500
                )
                results = []
                for feature, weight in exp.as_list(label=1):
                    results.append({
                        "Feature / condition":feature,
                        "LIME weight":weight,
                        "Direction":"Supports Diabetes" if weight > 0 else "Supports No Diabetes"
                    })
                st.session_state.lime_result = (exp, pd.DataFrame(results))

        if st.session_state.lime_result is not None:
            exp, lime_df = st.session_state.lime_result
            st.markdown("### Local LIME evidence")
            c1,c2 = st.columns([1.35,1])
            with c1:
                fig = exp.as_pyplot_figure(label=1)
                st.pyplot(fig, use_container_width=True)
            with c2:
                st.dataframe(lime_df, use_container_width=True, hide_index=True)
                st.markdown("**Interpretation guide**")
                st.write("Positive weights indicate conditions associated with the Diabetes class in the local surrogate; negative weights indicate the opposite.")
            st.markdown("### Ranked local conditions")
            fig, ax = plt.subplots(figsize=(9,5.5))
            d = lime_df.sort_values("LIME weight")
            ax.barh(d["Feature / condition"], d["LIME weight"])
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("LIME local weight")
            ax.set_title("Local surrogate feature contributions", fontweight="bold")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

# ============================================================
# COUNTERFACTUAL
# ============================================================
elif page == "🔄 Counterfactual":
    st.markdown("## Counterfactual explanation")
    st.caption("Counterfactuals answer: which allowed feature change can make the model output the opposite class? This is a model-behavior analysis, not a treatment recommendation.")

    if st.session_state.input_df is None:
        st.info("Run a prediction first.")
    else:
        original = st.session_state.input_df.iloc[0].copy()
        original_prediction = int(st.session_state.prediction)
        candidate_features = st.multiselect(
            "Features allowed to change",
            feature_columns,
            default=[f for f in ["BMI","glucose","systolic_avg","diastolic_avg","Hemoglobin level","age"] if f in feature_columns]
        )
        max_changes = st.slider("Maximum standardized change searched per feature", 0.1, 5.0, 1.0, 0.1)
        step = st.slider("Search step", 0.02, 0.50, 0.10, 0.01)

        st.info(f"Current model class: **{'Diabetes' if original_prediction else 'No Diabetes'}**")

        if st.button("Find nearest one-feature counterfactual", type="primary"):
            found = None
            for feature in candidate_features:
                old = float(original[feature])
                values = np.arange(old-max_changes, old+max_changes+step, step)
                # Search closest values first.
                values = sorted(values, key=lambda x: abs(x-old))
                for nv in values:
                    candidate = original.copy()
                    candidate[feature] = nv
                    candidate_df = pd.DataFrame([candidate], columns=feature_columns).astype(float)
                    pred = int(stack_nn.predict(candidate_df)[0])
                    prob = float(stack_nn.predict_proba(candidate_df)[0,1])
                    if pred != original_prediction:
                        found = (feature, old, nv, pred, prob, candidate_df)
                        break
                if found: break

            if found is None:
                st.warning("No one-feature counterfactual was found within the selected search range. Increase the search range or allow additional features.")
            else:
                feature, old, nv, pred, prob, candidate_df = found
                st.session_state.counterfactual = found

        if st.session_state.counterfactual is not None:
            feature, old, nv, pred, prob, candidate_df = st.session_state.counterfactual
            orig_prob = float(st.session_state.probabilities[1])
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Changed feature", feature)
            c2.metric("Original", f"{old:.3f}")
            c3.metric("Counterfactual", f"{nv:.3f}", delta=f"{nv-old:+.3f}")
            c4.metric("Diabetes probability", f"{prob*100:.2f}%", delta=f"{(prob-orig_prob)*100:+.2f} pp")

            st.success(f"One-feature counterfactual found: changing **{feature}** from **{old:.3f}** to **{nv:.3f}** changes the model class to **{'Diabetes' if pred else 'No Diabetes'}**.")

            comparison = pd.DataFrame({
                "Feature":feature_columns,
                "Original processed value":original.values,
                "Counterfactual value":candidate_df.iloc[0].values
            })
            comparison["Change"] = comparison["Counterfactual value"] - comparison["Original processed value"]
            comparison["Changed"] = comparison["Feature"].eq(feature)
            st.dataframe(comparison[comparison["Changed"] | (comparison["Change"].abs() > 1e-12)], use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(8,4.5))
            labels = ["Original","Counterfactual"]
            vals = [orig_prob, prob]
            ax.bar(labels, vals)
            ax.set_ylim(0,1); ax.set_ylabel("Diabetes probability")
            ax.set_title("Prediction change after counterfactual modification", fontweight="bold")
            for i,v in enumerate(vals):
                ax.text(i,v+.025,f"{v*100:.1f}%",ha="center",fontweight="bold")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

# ============================================================
# PDP
# ============================================================
elif page == "📈 PDP Analysis":
    st.markdown("## Partial dependence analysis")
    st.caption("PDP estimates the average model response as a feature varies over the test distribution. It describes model behavior and should not be interpreted as a causal relationship.")

    selected_features = st.multiselect(
        "Select features",
        feature_columns,
        default=[f for f in ["glucose","BMI","age","Hemoglobin level"] if f in feature_columns][:3]
    )
    grid_resolution = st.slider("Grid resolution", 10, 50, 25, 5)

    if selected_features:
        if st.button("Generate PDP analysis", type="primary"):
            with st.spinner("Computing partial dependence..."):
                for feature in selected_features:
                    fig, ax = plt.subplots(figsize=(9,5.2))
                    PartialDependenceDisplay.from_estimator(
                        stack_nn, X_test, [feature],
                        response_method="predict_proba",
                        target=1, grid_resolution=grid_resolution,
                        ax=ax
                    )
                    ax.set_title(f"Partial dependence: {feature}", fontweight="bold")
                    ax.set_ylabel("Average predicted probability of Diabetes")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
    else:
        st.info("Select at least one feature.")

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "📋 Model Performance":
    st.markdown("## Model performance — held-out test set")
    st.caption("Only held-out test-set results are presented here. Training-set performance is intentionally excluded to avoid presenting potentially misleading near-perfect training scores.")

    metric_cards({
        "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
        "Precision": f"{metrics['Precision']*100:.2f}%",
        "Sensitivity": f"{metrics['Recall / Sensitivity']*100:.2f}%",
        "Specificity": f"{metrics['Specificity']*100:.2f}%",
        "F1": f"{metrics['F1 Score']*100:.2f}%",
        "ROC-AUC": f"{metrics['ROC-AUC']:.3f}"
    })

    tabs = st.tabs(["Metric comparison","ROC curve","Precision–Recall","Confusion matrix","Calibration","Probability distribution"])

    with tabs[0]:
        display = model_compare.copy()
        st.dataframe(display.style.format({
            "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}","ROC-AUC":"{:.3f}"
        }), use_container_width=True, hide_index=True)

        plot_long = model_compare.set_index("Model")[["Accuracy","Precision","Recall","F1","ROC-AUC"]]
        fig, ax = plt.subplots(figsize=(10,5.5))
        plot_long.plot(kind="bar", ax=ax)
        ax.set_ylim(0,1.05); ax.set_ylabel("Score")
        ax.set_title("Held-out test-set performance comparison", fontweight="bold")
        ax.legend(ncol=3)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(8,6))
        # Use test predictions from each model.
        for name, mdl in [
            ("Extra Trees", stack_nn.named_estimators_["et"]),
            ("XGBoost", stack_nn.named_estimators_["xgb"]),
            ("Stacking Ensemble", stack_nn)
        ]:
            p = mdl.predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve(y_test,p)
            ax.plot(fpr,tpr,label=f"{name} (AUC={roc_auc_score(y_test,p):.3f})")
        ax.plot([0,1],[0,1],"--",linewidth=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves — held-out test set", fontweight="bold")
        ax.legend(); fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tabs[2]:
        fig, ax = plt.subplots(figsize=(8,6))
        for name, mdl in [
            ("Extra Trees", stack_nn.named_estimators_["et"]),
            ("XGBoost", stack_nn.named_estimators_["xgb"]),
            ("Stacking Ensemble", stack_nn)
        ]:
            p = mdl.predict_proba(X_test)[:,1]
            precision,recall,_ = precision_recall_curve(y_test,p)
            ax.plot(recall,precision,label=name)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall curves — held-out test set", fontweight="bold")
        ax.legend(); fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    with tabs[3]:
        st.pyplot(plot_confusion(confusion), use_container_width=True)
        st.dataframe(pd.DataFrame(class_report).T, use_container_width=True)

    with tabs[4]:
        fig, ax = plt.subplots(figsize=(8,6))
        frac, mean_pred = calibration_curve(y_test,test_probabilities,n_bins=10,strategy="uniform")
        ax.plot(mean_pred,frac,"o-",label="Stacking ensemble")
        ax.plot([0,1],[0,1],"--",linewidth=1,label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction positive")
        ax.set_title(f"Calibration curve (Brier score={metrics['Brier Score']:.4f})",fontweight="bold")
        ax.legend(); fig.tight_layout(); st.pyplot(fig,use_container_width=True)

    with tabs[5]:
        st.pyplot(probability_histogram(y_test,test_probabilities), use_container_width=True)

# ============================================================
# ARCHITECTURE
# ============================================================
elif page == "🧩 Model Architecture":
    st.markdown("## Stacking ensemble architecture")
    st.markdown("""
    <div class="hero">
      <b>Input features</b> → <b>Extra Trees</b> + <b>XGBoost</b> → probability-level meta-features → <b>MLP meta learner (64 → 32 → 16)</b> → final class probability
    </div>
    """, unsafe_allow_html=True)

    architecture_df = pd.DataFrame({
        "Component":["Base learner 1","Base learner 2","Stacking method","Passthrough","Meta learner","Hidden layers","Activation","Solver","Alpha","Batch size","Learning rate"],
        "Specification":["Extra Trees","XGBoost","predict_proba","False","MLPClassifier","64 → 32 → 16","ReLU","Adam","0.001","64","0.001"]
    })
    st.dataframe(architecture_df,use_container_width=True,hide_index=True)

    st.markdown("### Feature set")
    st.dataframe(pd.DataFrame({"Feature":feature_columns}),use_container_width=True,hide_index=True)

    st.markdown("### Model comparison on the held-out test set")
    st.dataframe(model_compare.style.format({
        "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}","ROC-AUC":"{:.3f}"
    }),use_container_width=True,hide_index=True)

# ============================================================
# RESEARCH NOTES
# ============================================================
elif page == "ℹ️ Research Notes":
    st.markdown("## Research interpretation and reporting notes")
    st.markdown("""
    ### Machine-learning model
    The dashboard implements the supplied stacking design: Extra Trees and XGBoost are the base learners, while an MLP neural network acts as the meta learner using probability outputs.

    ### Explainable AI
    - **SHAP:** local additive contribution analysis, with optional global summaries from a controlled test sample.
    - **LIME:** local surrogate explanation around the selected observation.
    - **Counterfactual:** searches for a one-feature change that flips the model class within a user-defined standardized range.
    - **PDP:** describes the average model response while varying selected features.

    ### Performance reporting
    The performance page reports only held-out test-set metrics and diagnostic plots. Training-set scores are deliberately not displayed because they can be substantially higher and can distract from generalization performance.

    ### Research caution
    XAI methods explain model behavior; they do not establish causality. Counterfactual changes are mathematical/model-based scenarios and should not be interpreted as clinical recommendations.
    """)
    st.markdown("### Classes")
    st.write("Class 0 = No Diabetes · Class 1 = Diabetes")
    st.markdown("""
    <div class="disclaimer"><b>Research/educational use:</b> this system is not a substitute for professional medical diagnosis, treatment or clinical decision-making.</div>
    """, unsafe_allow_html=True)
