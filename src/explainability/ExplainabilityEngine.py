import joblib
from pathlib import Path


class ExplainabilityEngine:

    def __init__(self):

        print("ExplainabilityEngine initialized.")

        # ====================================================
        # MODEL DIRECTORY
        # ====================================================

        self.model_directory = (
            Path(__file__).resolve()
            .parents[3]
            / "models"
        )

        self.model_path = (
            self.model_directory
            / "random_forest_pipeline.pkl"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Trained model not found: "
                f"{self.model_path}"
            )

        model = joblib.load(
            self.model_path
        )

        print(
            f"Model loaded successfully:\n"
            f"{self.model_path}"
        )

        return model

    # ========================================================
    # GENERATE CUSTOMER-SPECIFIC EXPLANATION
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

        model = self.load_model()

        # ----------------------------------------------------
        # Extract decision information
        # ----------------------------------------------------

        prediction = (
            decision_result["prediction"]
        )

        probability = (
            decision_result["probability_percent"]
        )

        risk = (
            decision_result["risk_category"]
        )

        explanations = []

        # ====================================================
        # PREDICTION EXPLANATION
        # ====================================================

        explanations.append(
            f"The model predicts {prediction} "
            f"with a probability of "
            f"{probability:.2f}%."
        )

        # ====================================================
        # RISK EXPLANATION
        # ====================================================

        explanations.append(
            f"The resulting model risk category "
            f"is {risk}."
        )

        # ====================================================
        # CUSTOMER-SPECIFIC FACTORS
        # ====================================================

        explanations.append(
            "The following customer attributes "
            "were considered when interpreting "
            "the prediction:"
        )

        # ----------------------------------------------------
        # Previous campaign history
        # ----------------------------------------------------

        previous = customer_data.get(
            "previous",
            0
        )

        if previous == 0:

            explanations.append(
                "The customer has no previous "
                "campaign contact history."
            )

        else:

            explanations.append(
                f"The customer has been contacted "
                f"{previous} time(s) in previous campaigns."
            )

        # ----------------------------------------------------
        # Previous campaign outcome
        # ----------------------------------------------------

        poutcome = customer_data.get(
            "poutcome",
            "unknown"
        )

        if poutcome == "success":

            explanations.append(
                "The customer has a successful "
                "previous campaign outcome, which "
                "provides a positive historical signal."
            )

        elif poutcome == "failure":

            explanations.append(
                "The customer has a previous failed "
                "campaign outcome, which provides a "
                "negative historical signal."
            )

        else:

            explanations.append(
                "There is no known outcome from a "
                "previous campaign contact."
            )

        # ----------------------------------------------------
        # Campaign intensity
        # ----------------------------------------------------

        campaign = customer_data.get(
            "campaign",
            0
        )

        if campaign <= 2:

            explanations.append(
                f"The current campaign contact count "
                f"is {campaign}, indicating relatively "
                f"low contact intensity."
            )

        elif campaign <= 5:

            explanations.append(
                f"The current campaign contact count "
                f"is {campaign}, indicating moderate "
                f"contact intensity."
            )

        else:

            explanations.append(
                f"The current campaign contact count "
                f"is {campaign}, indicating high "
                f"contact intensity."
            )

        # ----------------------------------------------------
        # Contact channel
        # ----------------------------------------------------

        contact = customer_data.get(
            "contact",
            "unknown"
        )

        if contact == "cellular":

            explanations.append(
                "The customer was contacted through "
                "a cellular communication channel."
            )

        elif contact == "telephone":

            explanations.append(
                "The customer was contacted through "
                "a telephone communication channel."
            )

        else:

            explanations.append(
                "The communication channel is unknown."
            )

        # ----------------------------------------------------
        # Contact duration
        # ----------------------------------------------------

        duration = customer_data.get(
            "duration",
            0
        )

        if duration >= 300:

            explanations.append(
                f"The current contact duration is "
                f"{duration} seconds, indicating "
                f"relatively strong customer engagement."
            )

        elif duration >= 100:

            explanations.append(
                f"The current contact duration is "
                f"{duration} seconds, indicating "
                f"moderate customer engagement."
            )

        else:

            explanations.append(
                f"The current contact duration is "
                f"{duration} seconds, indicating "
                f"relatively limited engagement."
            )

        # ----------------------------------------------------
        # Age
        # ----------------------------------------------------

        age = customer_data.get(
            "age",
            None
        )

        age_group = customer_data.get(
            "age_group",
            None
        )

        if age is not None:

            if age_group:

                explanations.append(
                    f"The customer is {age} years old "
                    f"and belongs to the "
                    f"{age_group} age group."
                )

            else:

                explanations.append(
                    f"The customer is {age} years old."
                )

        # ----------------------------------------------------
        # Account balance
        # ----------------------------------------------------

        balance = customer_data.get(
            "balance",
            0
        )

        if balance > 0:

            explanations.append(
                f"The customer has a positive "
                f"account balance of {balance}."
            )

        elif balance < 0:

            explanations.append(
                f"The customer has a negative "
                f"account balance of {balance}."
            )

        else:

            explanations.append(
                "The customer has a zero account balance."
            )

        # ====================================================
        # GLOBAL MODEL FEATURE IMPORTANCE
        # ====================================================

        top_features = []

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

            feature_importance_pairs = sorted(
                zip(
                    feature_names,
                    importances
                ),
                key=lambda item: item[1],
                reverse=True
            )

            for feature, importance in (
                feature_importance_pairs[:10]
            ):

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

                top_features.append(
                    {
                        "feature":
                            clean_feature,

                        "importance":
                            float(importance)
                    }
                )

        except Exception as error:

            print(
                f"Feature importance generation "
                f"failed: {error}"
            )

        # ====================================================
        # DISPLAY EXPLANATIONS
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "              CUSTOMER-SPECIFIC EXPLANATION"
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
            "Top Global Model Features:"
        )

        for number, item in enumerate(
            top_features,
            start=1
        ):

            print(
                f"   {number}. "
                f"{item['feature']} "
                f"(importance: "
                f"{item['importance']:.4f})"
            )

        print()

        print(
            "=" * 70
        )

        return {

            "explanations":
                explanations,

            "top_features":
                top_features
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = ExplainabilityEngine()

    customer = {

        "age": 35,

        "balance": 1500,

        "campaign": 2,

        "previous": 0,

        "poutcome": "unknown",

        "contact": "cellular",

        "duration": 300,

        "age_group": "31-40"
    }

    decision_result = {

        "prediction": "NO",

        "probability": 0.4539,

        "probability_percent": 45.39,

        "risk_category": "MEDIUM",

        "priority": "MEDIUM",

        "recommended_action":
            "Include customer in standard "
            "marketing follow-up."
    }

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