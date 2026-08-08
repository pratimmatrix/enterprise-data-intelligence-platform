from src.modeling.ModelPredictor import ModelPredictor
from src.business_rules.BusinessRuleEngine import BusinessRuleEngine


class DecisionEngine:

    def __init__(self):

        print("DecisionEngine initialized.")

        self.predictor = ModelPredictor()

        self.business_rules = BusinessRuleEngine()

    # ========================================================
    # RUN DECISION
    # ========================================================

    def run(
        self,
        customer_data
    ):

        print()
        print("=" * 70)
        print("                  DECISION ENGINE")
        print("=" * 70)

        # ----------------------------------------------------
        # STEP 1: ML PREDICTION
        # ----------------------------------------------------

        print()
        print("Step 1: Generating ML prediction...")

        prediction = self.predictor.predict(
            customer_data
        )

        print(
            f"Prediction : "
            f"{prediction['prediction']}"
        )

        print(
            f"Probability: "
            f"{prediction['probability_percent']}%"
        )

        # ----------------------------------------------------
        # STEP 2: BUSINESS RULES
        # ----------------------------------------------------

        print()
        print("Step 2: Applying business rules...")

        business_decision = (
            self.business_rules.evaluate(
                prediction
            )
        )

        print(
            f"Priority: "
            f"{business_decision['priority']}"
        )

        print(
            f"Action: "
            f"{business_decision['recommended_action']}"
        )

        # ----------------------------------------------------
        # STEP 3: COMBINE RESULTS
        # ----------------------------------------------------

        final_result = {

            "prediction":
                prediction["prediction"],

            "probability":
                prediction["probability"],

            "probability_percent":
                prediction["probability_percent"],

            "risk_category":
                prediction["risk_category"],

            "priority":
                business_decision["priority"],

            "recommended_action":
                business_decision[
                    "recommended_action"
                ]
        }

        return final_result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = DecisionEngine()

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
    # Run complete decision
    # --------------------------------------------------------

    result = engine.run(
        customer
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("                    FINAL DECISION")
    print("=" * 70)

    print()

    print(
        f"Prediction        : "
        f"{result['prediction']}"
    )

    print(
        f"Probability       : "
        f"{result['probability_percent']}%"
    )

    print(
        f"Risk Category     : "
        f"{result['risk_category']}"
    )

    print(
        f"Priority          : "
        f"{result['priority']}"
    )

    print(
        f"Recommended Action: "
        f"{result['recommended_action']}"
    )

    print()
    print("=" * 70)