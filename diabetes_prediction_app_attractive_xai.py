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

.stApp {
    background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 42%, #f8fafc 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #e6eaf0;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e7ebf0;
    padding: 14px 16px;
    border-radius: 14px;
    box-shadow: 0 3px 14px rgba(20, 40, 80, 0.06);
}

.xai-card {
    background: #ffffff;
    border: 1px solid #e6ebf1;
    border-radius: 16px;
    padding: 18px 20px;
    min-height: 105px;
    box-shadow: 0 4px 16px rgba(20, 40, 80, 0.05);
}

.xai-card h4 {
    margin: 0 0 7px 0;
}

.section-note {
    background: #f5f8fc;
    border-left: 4px solid #4f7cff;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0 18px 0;
}

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
# SESSION STATE
# ============================================================

if "input_df" not in st.session_state:

    st.session_state.input_df = None

if "prediction" not in st.session_state:

    st.session_state.prediction = None

if "probabilities" not in st.session_state:

    st.session_state.probabilities = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧭 Navigation"
)

page = st.sidebar.radio(

    "Select Section",

    [

        "🏠 Dashboard",

        "🔮 Prediction",

        "📊 SHAP",

        "🍋 LIME",

        "🔄 Counterfactual",

        "📈 PDP",

        "💬 XAI Chat",

        "📋 Model Performance",

        "🧩 Architecture",

        "ℹ️ About"

    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">Diabetes Prediction & Explainable AI</div>'
        '<div class="subtitle">Stacking ensemble model with SHAP, LIME, counterfactual and PDP analysis</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Test Accuracy", f"{metrics['Testing Accuracy']*100:.2f}%")
    with c2:
        st.metric("Test F1 Score", f"{metrics['Testing F1']*100:.2f}%")
    with c3:
        st.metric("Test ROC-AUC", f"{metrics['Testing ROC-AUC']*100:.2f}%")
    with c4:
        st.metric("Model Features", len(feature_columns))

    st.markdown(
        '<div class="section-note"><b>Research evaluation:</b> Performance values shown here are from the held-out test set. Training performance is not displayed.</div>',
        unsafe_allow_html=True
    )

    st.subheader("Model Overview")

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.markdown(
            '<div class="xai-card"><h4>🧩 Stacking Ensemble</h4>'
            '<p><b>Extra Trees</b> + <b>XGBoost</b> provide base predictions, '
            'which are combined by an <b>MLP meta learner</b> (64 → 32 → 16).</p></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="xai-card"><h4>🔬 Explainability Suite</h4>'
            '<p>Inspect individual predictions and understand model behaviour '
            'from multiple XAI perspectives.</p></div>',
            unsafe_allow_html=True
        )

    st.subheader("Available XAI Analysis")

    cards = [
        ("📊 SHAP", "Global and local feature contribution analysis."),
        ("🍋 LIME", "Local interpretable feature-weight explanation."),
        ("🔄 Counterfactual", "Find a nearby input that changes the model output."),
        ("📈 PDP", "Visualize model response across feature values.")
    ]

    cols = st.columns(4)
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="xai-card"><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True
            )

    st.subheader("Test-Set Performance")

    perf_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "ROC-AUC"],
        "Value": [
            metrics["Testing Accuracy"],
            metrics["Testing F1"],
            metrics["Testing ROC-AUC"]
        ]
    })

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(perf_df["Metric"], perf_df["Value"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test-Set Performance")
    for i, v in enumerate(perf_df["Value"]):
        ax.text(i, min(v + 0.025, 0.98), f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.header(
        "🔮 Diabetes Prediction"
    )

    st.write(
        "Enter patient/sample information."
    )

    input_data = {}

    # --------------------------------------------------------
    # Numeric inputs
    # --------------------------------------------------------

    numeric_features = [

        "age",

        "education_level",

        "BMI",

        "smoking",

        "alcohol",

        "hypertension",

        "heart_disease",

        "bp_cuff_size",

        "glucose",

        "wealth_index",

        "systolic_avg",

        "diastolic_avg",

        "Hemoglobin level",

        "Anemia level"

    ]

    available_numeric = [

        f for f in numeric_features

        if f in feature_columns
    ]

    st.subheader(
        "Patient Features"
    )

    cols = st.columns(2)

    for i, feature in enumerate(
        available_numeric
    ):

        with cols[i % 2]:

            default = float(
                X_train[feature].median()
            )

            input_data[feature] = st.number_input(

                feature,

                value=default,

                format="%.4f"
            )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    st.subheader(
        "Categorical Variables"
    )

    if "residence_type_Urban" in feature_columns:

        residence = st.selectbox(
            "Residence",
            [
                "Urban",
                "Rural"
            ]
        )

        input_data[
            "residence_type_Urban"
        ] = int(
            residence == "Urban"
        )

        input_data[
            "residence_type_Rural"
        ] = int(
            residence == "Rural"
        )

    if "gender_Female" in feature_columns:

        gender = st.selectbox(

            "Gender",

            [
                "Female",
                "Male",
                "Other"
            ]
        )

        input_data[
            "gender_Female"
        ] = int(
            gender == "Female"
        )

        input_data[
            "gender_Male"
        ] = int(
            gender == "Male"
        )

        input_data[
            "gender_Other"
        ] = int(
            gender == "Other"
        )

    marital_columns = [

        "marital_status_Divorced/Separated",

        "marital_status_Married",

        "marital_status_Never married",

        "marital_status_Widowed"

    ]

    if any(
        x in feature_columns
        for x in marital_columns
    ):

        marital = st.selectbox(

            "Marital Status",

            [
                "Married",
                "Never married",
                "Widowed",
                "Divorced/Separated"
            ]
        )

        for col in marital_columns:

            if col in feature_columns:

                input_data[col] = int(

                    col.replace(
                        "marital_status_",
                        ""
                    )
                    == marital

                )

    # --------------------------------------------------------
    # Missing columns
    # --------------------------------------------------------

    for feature in feature_columns:

        if feature not in input_data:

            input_data[feature] = float(
                X_train[feature].median()
            )

    # --------------------------------------------------------
    # Correct order
    # --------------------------------------------------------

    raw_input = pd.DataFrame(
        [input_data]
    )

    raw_input = raw_input[
        feature_columns
    ]

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in raw_input.columns:

        raw_input[col] = pd.to_numeric(
            raw_input[col],
            errors="coerce"
        )

    raw_input = raw_input.fillna(
        X_train.median()
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    input_scaled = raw_input.copy()

    if len(continuous_features) > 0:

        input_scaled[
            continuous_features
        ] = scaler.transform(
            input_scaled[
                continuous_features
            ]
        )

    input_scaled = input_scaled.astype(
        np.float64
    )

    with st.expander(
        "View final model input"
    ):

        st.dataframe(
            input_scaled,
            use_container_width=True
        )

    st.divider()

    if st.button(
        "🚀 Predict Diabetes",
        type="primary",
        use_container_width=True
    ):

        prediction = stack_nn.predict(
            input_scaled
        )[0]

        probabilities = stack_nn.predict_proba(
            input_scaled
        )[0]

        st.session_state.input_df = (
            input_scaled
        )

        st.session_state.prediction = (
            prediction
        )

        st.session_state.probabilities = (
            probabilities
        )

        st.subheader(
            "Prediction Result"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Prediction",
                "Diabetes"
                if prediction == 1
                else "No Diabetes"
            )

        with c2:

            st.metric(
                "Diabetes Probability",
                f"{probabilities[1]*100:.2f}%"
            )

        with c3:

            st.metric(
                "Confidence",
                f"{max(probabilities)*100:.2f}%"
            )

        if prediction == 1:

            st.error(
                "⚠️ Model prediction: Diabetes (Class 1)"
            )

        else:

            st.success(
                "✓ Model prediction: No Diabetes (Class 0)"
            )

        st.subheader(
            "Class Probabilities"
        )

        probability_df = pd.DataFrame({

            "Class": [
                "No Diabetes",
                "Diabetes"
            ],

            "Probability": probabilities,

            "Percentage":
                probabilities * 100

        })

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            probability_df.set_index(
                "Class"
            )[["Probability"]]
        )


# ============================================================
# SHAP
# ============================================================

elif page == "📊 SHAP":

    st.header("📊 SHAP Explainability")
    st.markdown(
        '<div class="section-note">Generate the complete SHAP analysis for the current prediction. '
        'These plots describe model behaviour and feature contribution.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.input_df is None:
        st.warning("Make a prediction first.")
    else:
        input_df = st.session_state.input_df

        if st.button("Generate Complete SHAP Analysis", type="primary"):

            with st.spinner("Calculating SHAP values and generating visualizations..."):

                def model_positive(X):
                    X = pd.DataFrame(
                        np.asarray(X, dtype=np.float64),
                        columns=feature_columns
                    ).astype(np.float64)
                    return stack_nn.predict_proba(X)[:, 1]

                background = X_train.sample(
                    min(50, len(X_train)), random_state=42
                ).values.astype(np.float64)

                explainer = shap.KernelExplainer(model_positive, background)

                local_values = explainer.shap_values(
                    input_df.values.astype(np.float64), nsamples=100
                )
                local_values = np.asarray(local_values, dtype=np.float64)
                if local_values.ndim == 3:
                    local_values = local_values[0, :, 0]
                elif local_values.ndim == 2:
                    local_values = local_values[0]
                else:
                    local_values = local_values.reshape(-1)
                local_values = np.nan_to_num(local_values)

                expected = float(
                    np.asarray(explainer.expected_value).reshape(-1)[0]
                )

                explanation = shap.Explanation(
                    values=local_values,
                    base_values=expected,
                    data=input_df.iloc[0].values,
                    feature_names=feature_columns
                )

                n_global = min(35, len(X_test))
                global_data = X_test.iloc[:n_global].values.astype(np.float64)
                global_values = explainer.shap_values(
                    global_data, nsamples=50
                )
                global_values = np.asarray(global_values, dtype=np.float64)
                if global_values.ndim == 3:
                    global_values = global_values[:, :, 0]
                global_values = np.nan_to_num(global_values)

            shap_df = pd.DataFrame({
                "Feature": feature_columns,
                "Value": input_df.iloc[0].values,
                "SHAP Value": local_values,
                "Absolute SHAP": np.abs(local_values)
            }).sort_values("Absolute SHAP", ascending=False)

            st.subheader("Local Feature Contributions")
            st.dataframe(
                shap_df[["Feature", "Value", "SHAP Value", "Absolute SHAP"]],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("SHAP Visualizations")
            tabs = st.tabs([
                "Contribution", "Bar", "Waterfall",
                "Beeswarm", "Dependence", "Heatmap"
            ])

            with tabs[0]:
                plot_df = shap_df.head(min(15, len(shap_df))).sort_values("SHAP Value")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(plot_df["Feature"], plot_df["SHAP Value"])
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("SHAP value")
                ax.set_title("Local SHAP Feature Contributions")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with tabs[1]:
                fig = plt.figure(figsize=(10, 6))
                shap.plots.bar(explanation, max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with tabs[2]:
                fig = plt.figure(figsize=(10, 7))
                shap.plots.waterfall(explanation, max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with tabs[3]:
                multi_exp = shap.Explanation(
                    values=global_values,
                    data=X_test.iloc[:len(global_values)].values,
                    feature_names=feature_columns
                )
                fig = plt.figure(figsize=(10, 7))
                shap.plots.beeswarm(multi_exp, max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with tabs[4]:
                selected_feature = st.selectbox(
                    "Feature for SHAP dependence plot",
                    feature_columns,
                    key="shap_dependence_feature"
                )
                feature_index = feature_columns.index(selected_feature)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.dependence_plot(
                    feature_index,
                    global_values,
                    X_test.iloc[:len(global_values)],
                    feature_names=feature_columns,
                    show=False,
                    ax=ax
                )
                ax.set_title(f"SHAP Dependence: {selected_feature}")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with tabs[5]:
                multi_exp = shap.Explanation(
                    values=global_values,
                    data=X_test.iloc[:len(global_values)].values,
                    feature_names=feature_columns
                )
                fig = plt.figure(figsize=(11, 7))
                shap.plots.heatmap(multi_exp, max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)


# ============================================================
# LIME
# ============================================================

elif page == "🍋 LIME":

    st.header("🍋 LIME Tabular Explanation")
    st.markdown(
        '<div class="section-note">LIME explains the current prediction locally by approximating the model around the selected sample.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.input_df is None:
        st.warning("Make a prediction first.")
    else:
        input_df = st.session_state.input_df

        if st.button("Generate LIME Analysis", type="primary"):

            with st.spinner("Generating LIME explanation..."):
                sample = input_df.iloc[0].values.astype(np.float64)
                explanation = lime_explainer.explain_instance(
                    sample,
                    stack_nn.predict_proba,
                    num_features=min(10, len(feature_columns)),
                    num_samples=3000
                )

            results = [
                {
                    "Feature / Condition": feature,
                    "Weight": weight,
                    "Direction": (
                        "Supports Diabetes" if weight > 0
                        else "Supports No Diabetes"
                    )
                }
                for feature, weight in explanation.as_list(label=1)
            ]

            lime_df = pd.DataFrame(results).sort_values("Weight", ascending=True)

            left, right = st.columns([1.1, 1])

            with left:
                st.subheader("LIME Contribution Plot")
                fig = explanation.as_pyplot_figure(label=1)
                fig.set_size_inches(9, 6)
                st.pyplot(fig, use_container_width=True)

            with right:
                st.subheader("LIME Weights")
                st.dataframe(lime_df, use_container_width=True, hide_index=True)

            st.subheader("LIME Feature Effects")
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.barh(lime_df["Feature / Condition"], lime_df["Weight"])
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("LIME weight")
            ax.set_title("Local Feature Weights")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)


# ============================================================
# COUNTERFACTUAL
# ============================================================

elif page == "🔄 Counterfactual":

    st.header("🔄 Counterfactual Explanation")
    st.markdown(
        '<div class="section-note">A counterfactual shows a nearby input change that can switch the model prediction.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.input_df is None:
        st.warning("Make a prediction first.")
    else:
        original = st.session_state.input_df.iloc[0].copy()
        original_prediction = int(st.session_state.prediction)
        original_probability = float(st.session_state.probabilities[1])

        candidate_features = st.multiselect(
            "Features allowed to change",
            feature_columns,
            default=[
                f for f in [
                    "BMI", "glucose", "systolic_avg",
                    "diastolic_avg", "Hemoglobin level", "age"
                ] if f in feature_columns
            ]
        )

        step = st.slider("Search step size", 0.01, 1.0, 0.10)
        max_changes = st.slider("Maximum change per feature", 0.1, 5.0, 1.0)

        if st.button("Find Counterfactual", type="primary"):

            with st.spinner("Searching for a counterfactual..."):
                found = None

                for feature in candidate_features:
                    original_value = float(original[feature])
                    values = np.arange(
                        original_value - max_changes,
                        original_value + max_changes + step,
                        step
                    )

                    for new_value in values:
                        candidate = original.copy()
                        candidate[feature] = new_value
                        candidate_df = pd.DataFrame(
                            [candidate], columns=feature_columns
                        ).astype(np.float64)

                        candidate_prediction = int(stack_nn.predict(candidate_df)[0])

                        if candidate_prediction != original_prediction:
                            candidate_probability = float(
                                stack_nn.predict_proba(candidate_df)[0, 1]
                            )
                            found = (
                                feature, original_value, new_value,
                                candidate_prediction, candidate_probability,
                                candidate_df
                            )
                            break

                    if found is not None:
                        break

            if found is None:
                st.warning("No counterfactual was found within the selected search range. Try increasing the maximum change.")
            else:
                (
                    feature, old_value, new_value,
                    new_prediction, new_probability, candidate_df
                ) = found

                st.success("Counterfactual found")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Changed Feature", feature)
                with c2:
                    st.metric("Original Value", f"{old_value:.4f}")
                with c3:
                    st.metric("Counterfactual Value", f"{new_value:.4f}")
                with c4:
                    st.metric("New Diabetes Probability", f"{new_probability*100:.2f}%")

                st.subheader("Changed Feature")
                comparison = pd.DataFrame({
                    "Feature": feature_columns,
                    "Original": original.values,
                    "Counterfactual": candidate_df.iloc[0].values
                })
                comparison["Difference"] = (
                    comparison["Counterfactual"] - comparison["Original"]
                )
                changed = comparison[
                    np.abs(comparison["Difference"].astype(float)) > 1e-12
                ]
                st.dataframe(changed, use_container_width=True, hide_index=True)

                st.subheader("Probability Change")
                prob_df = pd.DataFrame({
                    "Scenario": ["Original", "Counterfactual"],
                    "Diabetes Probability": [
                        original_probability * 100,
                        new_probability * 100
                    ]
                })

                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.bar(prob_df["Scenario"], prob_df["Diabetes Probability"])
                ax.set_ylabel("Diabetes probability (%)")
                ax.set_ylim(0, 100)
                ax.set_title("Original vs Counterfactual Prediction")
                for i, v in enumerate(prob_df["Diabetes Probability"]):
                    ax.text(i, min(v + 3, 98), f"{v:.1f}%", ha="center", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

                st.info(
                    f"The model prediction changed from **{original_prediction}** to **{new_prediction}** "
                    f"after changing **{feature}** within the selected search range."
                )


# ============================================================
# PDP
# ============================================================

elif page == "📈 PDP":

    st.header("📈 Partial Dependence Plots")
    st.markdown(
        '<div class="section-note">PDP shows how the model response changes as a selected feature varies across the test data.</div>',
        unsafe_allow_html=True
    )

    selected_features = st.multiselect(
        "Select features",
        feature_columns,
        default=[f for f in ["glucose", "BMI"] if f in feature_columns]
    )

    grid_resolution = st.slider(
        "Grid resolution", min_value=10, max_value=50, value=25, step=5
    )

    if not selected_features:
        st.info("Select at least one feature.")
    elif st.button("Generate PDP Analysis", type="primary"):

        with st.spinner("Generating Partial Dependence Plots..."):
            for feature in selected_features:
                st.subheader(f"PDP — {feature}")

                fig, ax = plt.subplots(figsize=(9, 5.5))

                PartialDependenceDisplay.from_estimator(
                    stack_nn,
                    X_test,
                    [feature],
                    response_method="predict_proba",
                    target=1,
                    grid_resolution=grid_resolution,
                    ax=ax
                )

                ax.set_title(
                    f"Partial Dependence of Diabetes Probability on {feature}"
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

        st.success(f"Generated {len(selected_features)} PDP plot(s).")


# ============================================================
# XAI CHAT
# ============================================================

elif page == "💬 XAI Chat":

    st.header(
        "💬 XAI Interpretation Assistant"
    )

    st.write(
        """
        Ask questions about the current prediction and
        the available XAI results.
        """
    )

    if st.session_state.input_df is None:

        st.warning(
            "Make a prediction first."
        )

    else:

        question = st.text_input(
            "Ask about this prediction",
            placeholder=
            "Which features most influenced the prediction?"
        )

        if st.button(
            "Explain",
            type="primary"
        ):

            input_df = (
                st.session_state.input_df
            )

            prediction = (
                st.session_state.prediction
            )

            probabilities = (
                st.session_state.probabilities
            )

            # ------------------------------------------------
            # Local feature importance using Extra Trees
            # as a quick interpretable approximation
            # ------------------------------------------------

            et_importance = (
                stack_nn
                .named_estimators_[
                    "et"
                ]
                .feature_importances_
            )

            importance_df = pd.DataFrame({

                "Feature":
                    feature_columns,

                "Importance":
                    et_importance,

                "Value":
                    input_df.iloc[0].values

            })

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
            )

            top = importance_df.head(
                5
            )

            if question == "":

                st.info(
                    "Enter a question."
                )

            else:

                q = question.lower()

                if (
                    "important" in q
                    or "influence" in q
                    or "feature" in q
                ):

                    st.subheader(
                        "Most Important Features"
                    )

                    st.dataframe(
                        top,
                        use_container_width=True,
                        hide_index=True
                    )

                    for _, row in top.iterrows():

                        st.write(
                            f"**{row['Feature']}** "
                            f"has a model importance of "
                            f"{row['Importance']:.4f} "
                            f"with current value "
                            f"{row['Value']:.4f}."
                        )

                elif (
                    "prediction" in q
                    or "result" in q
                ):

                    if prediction == 1:

                        st.error(
                            f"""
                            The model predicts **Diabetes**.

                            Diabetes probability:
                            **{probabilities[1]*100:.2f}%**

                            Overall confidence:
                            **{max(probabilities)*100:.2f}%**
                            """
                        )

                    else:

                        st.success(
                            f"""
                            The model predicts
                            **No Diabetes**.

                            Diabetes probability:
                            **{probabilities[1]*100:.2f}%**

                            Overall confidence:
                            **{max(probabilities)*100:.2f}%**
                            """
                        )

                elif (
                    "glucose" in q
                ):

                    if "glucose" in feature_columns:

                        value = float(
                            input_df[
                                "glucose"
                            ].iloc[0]
                        )

                        st.write(
                            f"""
                            The current processed glucose
                            value supplied to the model is
                            **{value:.4f}**.

                            The effect of glucose should be
                            interpreted together with SHAP,
                            LIME, and PDP rather than using
                            the feature value alone.
                            """
                        )

                elif (
                    "counterfactual" in q
                ):

                    st.write(
                        """
                        Go to the **Counterfactual** section
                        to identify feature changes that can
                        switch the model prediction.
                        """
                    )

                elif (
                    "shap" in q
                ):

                    st.write(
                        """
                        Go to the **SHAP** section to view
                        local feature contributions,
                        waterfall plots, dependence plots,
                        beeswarm plots, and heatmaps.
                        """
                    )

                else:

                    st.write(
                        f"""
                        The current prediction is
                        **{'Diabetes' if prediction == 1 else 'No Diabetes'}**
                        with a diabetes probability of
                        **{probabilities[1]*100:.2f}%**.

                        You can ask about:

                        - important features
                        - prediction
                        - glucose
                        - SHAP
                        - counterfactual explanations
                        """
                    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📋 Model Performance":

    st.header(
        "📋 Model Performance"
    )

    # Only the requested research metrics are shown on the dashboard.
    # The underlying model/training code above is unchanged.
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Test Accuracy",
            f"{metrics['Testing Accuracy']*100:.2f}%"
        )

    with c2:
        st.metric(
            "Test F1 Score",
            f"{metrics['Testing F1']*100:.2f}%"
        )

    with c3:
        st.metric(
            "Test ROC-AUC",
            f"{metrics['Testing ROC-AUC']:.3f}"
        )

    st.divider()

    # --------------------------------------------------------
    # ROC-AUC Curve
    # --------------------------------------------------------

    st.subheader(
        "ROC-AUC Curve"
    )

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(
        y_test,
        test_probabilities
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Stacking Ensemble (AUC = {metrics['Testing ROC-AUC']:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve on Held-Out Test Data"
    )

    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()

    st.pyplot(
        fig
    )

    st.caption(
        "Performance shown here is calculated on the held-out test set. "
        "Training performance is intentionally not displayed."
    )

    st.divider()

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(
        confusion
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        "Test-Set Confusion Matrix"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "No Diabetes",
            "Diabetes"
        ]
    )

    ax.set_yticklabels(
        [
            "No Diabetes",
            "Diabetes"
        ]
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                confusion[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    st.pyplot(
        fig
    )


# ============================================================
# ARCHITECTURE
# ============================================================

elif page == "🧩 Architecture":

    st.header(
        "🧩 Stacking Model Architecture"
    )

    st.code(
        """
                    INPUT FEATURES
                         |
              +----------+----------+
              |                     |
              v                     v
        EXTRA TREES              XGBOOST
              |                     |
              +----------+----------+
                         |
                  PREDICT_PROBA
                         |
                         v
                 META FEATURES
                         |
                         v
                  MLP CLASSIFIER
                    64 neurons
                         |
                    32 neurons
                         |
                    16 neurons
                         |
                         v
                 FINAL PREDICTION
                         |
             +-----------+-----------+
             |                       |
             v                       v
        NO DIABETES              DIABETES
           Class 0                 Class 1
        """
    )

    architecture_df = pd.DataFrame({

        "Component": [

            "Base Learner 1",

            "Base Learner 2",

            "Stacking Method",

            "Passthrough",

            "Meta Learner",

            "Hidden Layer 1",

            "Hidden Layer 2",

            "Hidden Layer 3",

            "Activation",

            "Solver",

            "Alpha",

            "Batch Size",

            "Learning Rate"

        ],

        "Value": [

            "Extra Trees",

            "XGBoost",

            "predict_proba",

            "False",

            "MLPClassifier",

            "64",

            "32",

            "16",

            "ReLU",

            "Adam",

            "0.001",

            "64",

            "0.001"

        ]

    })

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Features"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": feature_columns
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.header(
        "ℹ️ About"
    )

    st.markdown(
        """
        ## Explainable Diabetes Prediction System

        This application combines ensemble machine learning
        with explainable AI techniques.

        ### Machine Learning

        **Extra Trees + XGBoost**

        are used as base learners.

        Their probability outputs are supplied to an

        **MLP Neural Network**

        meta learner with architecture:

        **64 → 32 → 16**

        ### XAI Methods

        **SHAP**

        Explains individual feature contributions.

        **LIME**

        Provides a local interpretable approximation.

        **Counterfactual**

        Searches for feature changes that can alter
        the model prediction.

        **PDP**

        Shows the relationship between feature values
        and model predictions.

        **XAI Chat**

        Provides a simple interactive interpretation
        layer over the model and explanations.

        ### Classes

        **0 = No Diabetes**

        **1 = Diabetes**

        This system is intended for research and educational
        purposes and should not be used as a substitute for
        professional medical diagnosis.
        """
    )