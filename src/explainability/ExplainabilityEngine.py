import joblib
from pathlib import Path


class ExplainabilityEngine:

    def __init__(self):

        print("ExplainabilityEngine initialized.")

        # ====================================================
        # MODEL LOCATION
        # ====================================================

        self.model_path = (
            Path(__file__).resolve()
            .parents[2]
            / "models"
            / "random_forest_pipeline.pkl"
        )

        # If the model is stored outside the project,
        # use the existing location as fallback.

        if not self.model_path.exists():

            self.model_path = (
                Path.home()
                / "Documents"
                / "models"
                / "random_forest_pipeline.pkl"
            )

        self.model = None

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        print(
            f"Model loaded successfully:\n"
            f"{self.model_path}"
        )

        return self.model

    # ========================================================
    # GET FEATURE IMPORTANCE
    # ========================================================

    def get_feature_importance(
        self,
        top_n=10
    ):

        if self.model is None:

            self.load_model()

        # ----------------------------------------------------
        # Get preprocessing pipeline
        # ----------------------------------------------------

        preprocessor = (
            self.model.named_steps[
                "preprocessor"
            ]
        )

        # ----------------------------------------------------
        # Get classifier
        # ----------------------------------------------------

        classifier = (
            self.model.named_steps[
                "classifier"
            ]
        )

        # ----------------------------------------------------
        # Check Random Forest
        # ----------------------------------------------------

        if not hasattr(
            classifier,
            "feature_importances_"
        ):

            raise TypeError(
                "Selected model does not "
                "support feature_importances_."
            )

        # ----------------------------------------------------
        # Get transformed feature names
        # ----------------------------------------------------

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        # ----------------------------------------------------
        # Get importance values
        # ----------------------------------------------------

        importances = (
            classifier.feature_importances_
        )

        # ----------------------------------------------------
        # Combine feature + importance
        # ----------------------------------------------------

        feature_importance = list(
            zip(
                feature_names,
                importances
            )
        )

        # ----------------------------------------------------
        # Sort descending
        # ----------------------------------------------------

        feature_importance.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return feature_importance[:top_n]

    # ========================================================
    # GENERATE EXPLANATION
    # ========================================================

    def generate_explanation(
        self,
        customer_data,
        decision_result
    ):

        if customer_data is None:

            raise ValueError(
                "Customer data cannot be None."
            )

        if decision_result is None:

            raise ValueError(
                "Decision result cannot be None."
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        if self.model is None:

            self.load_model()

        prediction = (
            decision_result[
                "prediction"
            ]
        )

        probability = (
            decision_result[
                "probability_percent"
            ]
        )

        risk = (
            decision_result[
                "risk_category"
            ]
        )

        # ----------------------------------------------------
        # Get feature importance
        # ----------------------------------------------------

        top_features = (
            self.get_feature_importance(
                top_n=10
            )
        )

        explanations = []

        # ====================================================
        # MODEL RESULT
        # ====================================================

        explanations.append(
            f"The model predicts "
            f"{prediction} with a probability "
            f"of {probability:.2f}%."
        )

        explanations.append(
            f"The resulting model risk category "
            f"is {risk}."
        )

        # ====================================================
        # TOP MODEL FEATURES
        # ====================================================

        explanations.append(
            "The Random Forest considers the "
            "following features most important "
            "for its predictions:"
        )

        for index, (
            feature,
            importance
        ) in enumerate(
            top_features,
            start=1
        ):

            # Remove sklearn transformer prefixes
            clean_feature = (
                feature
                .replace(
                    "numeric__",
                    ""
                )
                .replace(
                    "categorical__",
                    ""
                )
            )

            explanations.append(
                f"{index}. "
                f"{clean_feature} "
                f"(importance: "
                f"{importance:.4f})"
            )

        # ====================================================
        # CUSTOMER-SPECIFIC INFORMATION
        # ====================================================

        if "age" in customer_data:

            explanations.append(
                f"Customer age: "
                f"{customer_data['age']}."
            )

        if "balance" in customer_data:

            balance = (
                customer_data[
                    "balance"
                ]
            )

            if balance > 0:

                explanations.append(
                    f"Customer has a positive "
                    f"account balance of "
                    f"{balance}."
                )

            elif balance < 0:

                explanations.append(
                    f"Customer has a negative "
                    f"account balance of "
                    f"{balance}."
                )

            else:

                explanations.append(
                    "Customer has a zero "
                    "account balance."
                )

        if "campaign" in customer_data:

            explanations.append(
                f"Current campaign contact "
                f"count: "
                f"{customer_data['campaign']}."
            )

        if "previous" in customer_data:

            explanations.append(
                f"Previous campaign contacts: "
                f"{customer_data['previous']}."
            )

        if "poutcome" in customer_data:

            explanations.append(
                f"Previous campaign outcome: "
                f"{customer_data['poutcome']}."
            )

        if "contact" in customer_data:

            explanations.append(
                f"Current contact channel: "
                f"{customer_data['contact']}."
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "                 MODEL EXPLANATION"
        )

        print(
            "=" * 70
        )

        print()

        for number, explanation in enumerate(
            explanations,
            start=1
        ):

            print(
                f"{number}. {explanation}"
            )

        print()

        print(
            "=" * 70
        )

        return {
            "explanations": explanations,
            "top_features": [
                {
                    "feature": feature,
                    "importance": float(
                        importance
                    )
                }
                for feature, importance
                in top_features
            ]
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = ExplainabilityEngine()

    # --------------------------------------------------------
    # Example customer
    # --------------------------------------------------------

    customer = {

        "age": 35,

        "job": "management",

        "marital": "married",

        "education": "tertiary",

        "default": "no",

        "balance": 1500,

        "housing": "yes",

        "loan": "no",

        "contact": "cellular",

        "day": 15,

        "month": "may",

        "duration": 300,

        "campaign": 2,

        "pdays": -1,

        "previous": 0,

        "poutcome": "unknown",

        "age_group": "31-40",

        "balance_log": 7.313,

        "campaign_log": 1.099,

        "previous_contact": 0,

        "previously_contacted": 0,

        "zero_balance": 0,

        "loan_burden": 0,

        "campaign_intensity": "low",

        "contact_unknown": 0,

        "previous_success": 0
    }

    # --------------------------------------------------------
    # Example decision result
    # --------------------------------------------------------

    decision_result = {

        "prediction": "NO",

        "probability": 0.4539,

        "probability_percent": 45.39,

        "risk_category": "MEDIUM",

        "priority": "LOW",

        "recommended_action":
            "Do not prioritize customer "
            "for immediate campaign outreach."
    }

    # --------------------------------------------------------
    # Generate explanation
    # --------------------------------------------------------

    result = (
        engine.generate_explanation(
            customer,
            decision_result
        )
    )

    print()

    print(
        "Explanation generation completed."
    )