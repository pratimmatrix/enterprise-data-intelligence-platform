import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Data Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    Path.home()
    / "Documents"
    / "models"
    / "random_forest_pipeline.pkl"
)

DATA_PATH = PROJECT_ROOT / "bank-full.csv"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            f"Model not found:\n{MODEL_PATH}"
        )

        st.stop()

    return joblib.load(MODEL_PATH)


model = load_model()

# ============================================================
# PAGE TITLE
# ============================================================

st.title("📊 Enterprise Data Intelligence Platform")

st.caption(
    "AI-powered customer campaign decision intelligence"
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Customer Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ],
    index=4
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["married", "single", "divorced"]
)

education = st.sidebar.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"],
    index=2
)

default = st.sidebar.selectbox(
    "Credit Default",
    ["no", "yes"]
)

balance = st.sidebar.number_input(
    "Account Balance",
    value=1500
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    ["yes", "no"]
)

loan = st.sidebar.selectbox(
    "Personal Loan",
    ["yes", "no"]
)

contact = st.sidebar.selectbox(
    "Contact Channel",
    ["cellular", "telephone", "unknown"],
    index=0
)

day = st.sidebar.number_input(
    "Contact Day",
    min_value=1,
    max_value=31,
    value=15
)

month = st.sidebar.selectbox(
    "Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
    ],
    index=4
)

duration = st.sidebar.number_input(
    "Contact Duration (seconds)",
    min_value=0,
    value=300
)

campaign = st.sidebar.number_input(
    "Campaign Contacts",
    min_value=1,
    value=2
)

pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    value=-1
)

previous = st.sidebar.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    [
        "unknown",
        "failure",
        "other",
        "success"
    ]
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features():

    customer = {

        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,

        "age_group": (
            "18-30" if age <= 30
            else "31-40" if age <= 40
            else "41-50" if age <= 50
            else "51-60" if age <= 60
            else "61-70" if age <= 70
            else "71+"
        ),

        "balance_log": (
            __import__("numpy").log1p(
                abs(balance)
            )
        ),

        "campaign_log": (
            __import__("numpy").log1p(
                campaign
            )
        ),

        "previous_contact": (
            1 if previous > 0 else 0
        ),

        "previously_contacted": (
            1 if pdays != -1 else 0
        ),

        "zero_balance": (
            1 if balance == 0 else 0
        ),

        "loan_burden": (
            1 if loan == "yes" else 0
        ),

        "campaign_intensity": (
            "low" if campaign <= 2
            else "medium" if campaign <= 5
            else "high"
        ),

        "contact_unknown": (
            1 if contact == "unknown" else 0
        ),

        "previous_success": (
            1 if poutcome == "success" else 0
        )
    }

    return pd.DataFrame([customer])


# ============================================================
# PREDICTION
# ============================================================

st.subheader("Customer Decision")

if st.button(
    "🚀 Generate Decision",
    use_container_width=True
):

    customer_df = create_features()

    try:

        prediction = model.predict(
            customer_df
        )[0]

        probability = model.predict_proba(
            customer_df
        )[0][1]

        probability_percent = (
            probability * 100
        )

        # ----------------------------------------------------
        # RISK CATEGORY
        # ----------------------------------------------------

        if probability_percent >= 70:

            risk = "HIGH"

        elif probability_percent >= 40:

            risk = "MEDIUM"

        else:

            risk = "LOW"

        # ----------------------------------------------------
        # BUSINESS PRIORITY
        # ----------------------------------------------------

        if risk == "HIGH":

            priority = "HIGH"

            action = (
                "Prioritize customer for immediate "
                "campaign outreach."
            )

        elif risk == "MEDIUM":

            priority = "MEDIUM"

            action = (
                "Include customer in standard "
                "marketing follow-up."
            )

        else:

            priority = "LOW"

            action = (
                "Do not prioritize customer for "
                "immediate campaign outreach."
            )

        # ----------------------------------------------------
        # DISPLAY METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Prediction",
                str(prediction).upper()
            )

        with col2:

            st.metric(
                "Probability",
                f"{probability_percent:.2f}%"
            )

        with col3:

            st.metric(
                "Risk Category",
                risk
            )

        with col4:

            st.metric(
                "Priority",
                priority
            )

        st.divider()

        # ====================================================
        # BUSINESS DECISION
        # ====================================================

        st.subheader(
            "💼 Business Decision"
        )

        if priority == "HIGH":

            st.success(action)

        elif priority == "MEDIUM":

            st.warning(action)

        else:

            st.info(action)

        # ====================================================
        # BUSINESS INSIGHTS
        # ====================================================

        st.subheader(
            "💡 Business Insights"
        )

        if prediction == "yes":

            st.write(
                "• Customer shows a positive likelihood "
                "of responding to the campaign."
            )

        else:

            st.write(
                "• Customer shows a lower likelihood "
                "of responding to the campaign."
            )

        if probability_percent >= 70:

            st.write(
                "• Prediction probability is high, "
                "indicating a strong model signal."
            )

        elif probability_percent >= 40:

            st.write(
                "• Prediction probability is moderate, "
                "indicating some potential for customer response."
            )

        else:

            st.write(
                "• Prediction probability is low, "
                "indicating a weak likelihood of customer response."
            )

        st.write(
            f"• Risk category is classified as **{risk}**."
        )

        # ====================================================
        # CUSTOMER PROFILE
        # ====================================================

        st.subheader(
            "👤 Customer Profile"
        )

        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:

            st.write(
                f"**Age:** {age}"
            )

            st.write(
                f"**Job:** {job}"
            )

            st.write(
                f"**Education:** {education}"
            )

            st.write(
                f"**Balance:** {balance}"
            )

            st.write(
                f"**Housing Loan:** {housing}"
            )

        with profile_col2:

            st.write(
                f"**Campaign Contacts:** {campaign}"
            )

            st.write(
                f"**Previous Contacts:** {previous}"
            )

            st.write(
                f"**Previous Outcome:** {poutcome}"
            )

            st.write(
                f"**Contact Channel:** {contact}"
            )

            st.write(
                f"**Duration:** {duration} seconds"
            )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.subheader(
            "🤖 Model Information"
        )

        st.write(
            "Model: **Random Forest Classifier**"
        )

        st.write(
            "Pipeline: **Preprocessor + Random Forest**"
        )

        st.write(
            "Selected using **F1 Score**"
        )

        st.write(
            "ROC-AUC: **0.7946**"
        )

        st.write(
            "F1 Score: **0.4297**"
        )

        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        st.subheader(
            "📈 Top Model Features"
        )

        try:

            preprocessor = (
                model.named_steps[
                    "preprocessor"
                ]
            )

            classifier = (
                model.named_steps[
                    "classifier"
                ]
            )

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            importances = (
                classifier.feature_importances_
            )

            importance_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Importance":
                    importances

            })

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
            )

            importance_df = (
                importance_df
                .set_index("Feature")
            )

            st.bar_chart(
                importance_df
            )

        except Exception as exc:

            st.warning(
                f"Feature importance unavailable: {exc}"
            )

        # ====================================================
        # EXPLANATION
        # ====================================================

        st.subheader(
            "🔍 Model Explanation"
        )

        st.write(
            f"The model predicts **{str(prediction).upper()}** "
            f"with a probability of "
            f"**{probability_percent:.2f}%**."
        )

        st.write(
            f"The resulting model risk category is "
            f"**{risk}**."
        )

        if previous == 0:

            st.write(
                "• The customer has no previous campaign "
                "contact history."
            )

        if poutcome == "unknown":

            st.write(
                "• There is no known outcome from a "
                "previous campaign contact."
            )

        st.write(
            f"• The customer has received "
            f"{campaign} campaign contact(s)."
        )

        st.write(
            f"• The customer was contacted through "
            f"a {contact} communication channel."
        )

        st.write(
            f"• Current contact duration is "
            f"{duration} seconds."
        )

        st.write(
            f"• Customer age is {age}."
        )

        st.write(
            f"• Customer account balance is {balance}."
        )

    except Exception as exc:

        st.error(
            "Prediction failed."
        )

        st.exception(exc)