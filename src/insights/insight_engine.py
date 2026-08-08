import logging
import pandas as pd

logger = logging.getLogger(__name__)


class InsightEngine:

    def __init__(self):
        print("InsightEngine initialized.")

    def analyze(self, df: pd.DataFrame) -> bool:

        print("\n========== INSIGHT INTELLIGENCE ==========")

        if df is None:
            raise ValueError("Input dataframe is None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if df.empty:
            raise ValueError("Dataset contains zero rows.")

        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        # --------------------------------------------------
        # TARGET ANALYSIS
        # --------------------------------------------------

        if "y" in df.columns:

            print("\n---------- TARGET INSIGHTS ----------")

            target_counts = df["y"].value_counts()

            for value, count in target_counts.items():

                percentage = (
                    count / len(df)
                ) * 100

                print(
                    f"{value:<10} "
                    f"{count:>8} records "
                    f"({percentage:.2f}%)"
                )

        # --------------------------------------------------
        # NUMERICAL INSIGHTS
        # --------------------------------------------------

        print("\n---------- NUMERICAL INSIGHTS ----------")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:

            mean = df[column].mean()
            median = df[column].median()

            print(
                f"{column:<20} "
                f"mean={mean:.2f} | "
                f"median={median:.2f}"
            )

        # --------------------------------------------------
        # CATEGORICAL TARGET INSIGHTS
        # --------------------------------------------------

        if "y" in df.columns:

            print(
                "\n---------- TOP CUSTOMER SEGMENTS ----------"
            )

            categorical_columns = df.select_dtypes(
                include=["object", "string", "category"]
            ).columns

            for column in categorical_columns:

                if column == "y":
                    continue

                try:

                    rates = (
                        df.groupby(column)["y"]
                        .apply(
                            lambda values:
                            (values == "yes").mean() * 100
                        )
                        .sort_values(
                            ascending=False
                        )
                    )

                    if not rates.empty:

                        top_category = rates.index[0]
                        top_rate = rates.iloc[0]

                        print(
                            f"{column:<15} "
                            f"{top_category} -> "
                            f"{top_rate:.2f}% target rate"
                        )

                except Exception as error:

                    logger.warning(
                        "Could not analyze %s: %s",
                        column,
                        error
                    )

        # --------------------------------------------------
        # FEATURE INSIGHTS
        # --------------------------------------------------

        engineered_features = [
            feature
            for feature in [
                "age_group",
                "balance_log",
                "campaign_log",
                "previous_contact",
                "previously_contacted",
                "zero_balance",
                "loan_burden",
                "campaign_intensity",
                "contact_unknown",
                "previous_success",
            ]
            if feature in df.columns
        ]

        print("\n---------- ENGINEERED FEATURE INSIGHTS ----------")

        print(
            f"Engineered features detected: "
            f"{len(engineered_features)}"
        )

        for feature in engineered_features:

            print(f"• {feature}")

        print("\nINSIGHT ANALYSIS PASSED")

        return True