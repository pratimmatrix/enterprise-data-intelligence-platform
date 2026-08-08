import pandas as pd


class ExplainabilityEngine:

    def __init__(self):

        print("ExplainabilityEngine initialized.")

    # ========================================================
    # GENERATE EXPLANATION
    # ========================================================

    def explain(
        self,
        customer_data,
        prediction_result
    ):

        if customer_data is None:

            raise ValueError(
                "Customer data cannot be None."
            )

        if prediction_result is None:

            raise ValueError(
                "Prediction result cannot be None."
            )

        # ----------------------------------------------------
        # Extract prediction information
        # ----------------------------------------------------

        prediction = (
            prediction_result["prediction"]
        )

        probability = (
            prediction_result["probability_percent"]
        )

        risk = (
            prediction_result["risk_category"]
        )

        # ----------------------------------------------------
        # Convert customer data to Series
        # ----------------------------------------------------

        customer = pd.Series(
            customer_data
        )

        explanations = []

        # ====================================================
        # CUSTOMER PROFILE
        # ====================================================

        age = customer.get(
            "age",
            None
        )

        balance = customer.get(
            "balance",
            None
        )

        campaign = customer.get(
            "campaign",
            None
        )

        previous = customer.get(
            "previous",
            None
        )

        poutcome = customer.get(
            "poutcome",
            None
        )

        contact = customer.get(
            "contact",
            None
        )

        duration = customer.get(
            "duration",
            None
        )

        # ====================================================
        # FACTOR 1: PREVIOUS CONTACT
        # ====================================================

        if previous == 0:

            explanations.append(
                "The customer has no previous campaign "
                "contact history."
            )

        elif previous > 0:

            explanations.append(
                f"The customer has been contacted "
                f"{previous} time(s) previously."
            )

        # ====================================================
        # FACTOR 2: PREVIOUS OUTCOME
        # ====================================================

        if poutcome == "success":

            explanations.append(
                "A previous campaign contact was successful, "
                "which is a positive engagement signal."
            )

        elif poutcome == "failure":

            explanations.append(
                "A previous campaign contact was unsuccessful, "
                "which is a negative engagement signal."
            )

        elif poutcome == "unknown":

            explanations.append(
                "There is no known outcome from a previous "
                "campaign contact."
            )

        # ====================================================
        # FACTOR 3: CAMPAIGN CONTACT FREQUENCY
        # ====================================================

        if campaign is not None:

            if campaign >= 5:

                explanations.append(
                    f"The customer has already been contacted "
                    f"{campaign} times in the current campaign, "
                    "indicating relatively high contact intensity."
                )

            elif campaign <= 2:

                explanations.append(
                    f"The customer has received only "
                    f"{campaign} campaign contact(s), "
                    "indicating relatively low contact intensity."
                )

            else:

                explanations.append(
                    f"The customer has received "
                    f"{campaign} campaign contacts."
                )

        # ====================================================
        # FACTOR 4: CONTACT METHOD
        # ====================================================

        if contact == "cellular":

            explanations.append(
                "The customer was contacted through a "
                "cellular communication channel."
            )

        elif contact == "telephone":

            explanations.append(
                "The customer was contacted through "
                "telephone communication."
            )

        elif contact == "unknown":

            explanations.append(
                "The customer's contact communication "
                "channel is unknown."
            )

        # ====================================================
        # FACTOR 5: CALL DURATION
        # ====================================================

        if duration is not None:

            if duration < 100:

                explanations.append(
                    "The current contact duration is relatively "
                    "short, indicating limited engagement."
                )

            elif duration >= 300:

                explanations.append(
                    "The current contact duration indicates "
                    "relatively strong customer engagement."
                )

        # ====================================================
        # FACTOR 6: BALANCE
        # ====================================================

        if balance is not None:

            if balance < 0:

                explanations.append(
                    "The customer has a negative account balance."
                )

            elif balance == 0:

                explanations.append(
                    "The customer currently has a zero account balance."
                )

            elif balance > 0:

                explanations.append(
                    "The customer has a positive account balance."
                )

        # ====================================================
        # FACTOR 7: AGE
        # ====================================================

        if age is not None:

            if age < 30:

                explanations.append(
                    "The customer belongs to a relatively young "
                    "age group."
                )

            elif age >= 60:

                explanations.append(
                    "The customer belongs to an older age group."
                )

            else:

                explanations.append(
                    "The customer is in a middle adult age group."
                )

        # ====================================================
        # MODEL RESULT
        # ====================================================

        if prediction == "YES":

            explanations.append(
                f"The model predicts a positive campaign response "
                f"with a probability of {probability:.2f}%."
            )

        else:

            explanations.append(
                f"The model predicts a negative campaign response "
                f"with a probability of {probability:.2f}%."
            )

        # ====================================================
        # RISK
        # ====================================================

        explanations.append(
            f"The resulting model risk category is {risk}."
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        print()

        print("=" * 70)

        print(
            "                  MODEL EXPLANATION"
        )

        print("=" * 70)

        print()

        for number, explanation in enumerate(
            explanations,
            start=1
        ):

            print(
                f"{number}. {explanation}"
            )

        print()

        print("=" * 70)

        return {

            "explanations":
                explanations
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = ExplainabilityEngine()

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

    prediction = {

        "prediction": "NO",

        "probability": 0.3764,

        "probability_percent": 37.64,

        "risk_category": "LOW"
    }

    result = engine.explain(
        customer,
        prediction
    )

    print()

    print(
        "Explanation generation completed."
    )