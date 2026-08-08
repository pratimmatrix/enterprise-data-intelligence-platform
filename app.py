import logging
import sys
import pandas as pd

from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.profiling.metadata import MetadataExtractor
from src.profiling.profiler import DataProfiler
from src.profiling.quality import DataQualityEngine
from src.profiling.anomalies import AnomalyEngine


DATA_FILE = "data/raw/bank-full.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def section(title):
    print()
    print("=" * 70)
    print(f"{title:^70}")
    print("=" * 70)


def load_data():
    section("DATA INGESTION")

    logger.info("Starting data ingestion...")

    loader = DataLoader()
    df = loader.load(DATA_FILE)

    if df is None:
        raise RuntimeError("DataLoader returned no dataframe.")

    print()
    print("Data successfully loaded.")
    print(f"Dataset shape: {df.shape}")

    return df


def validate_data(df):
    section("DATA VALIDATION")

    validator = DataValidator()
    validator.validate(df)


def metadata_analysis(df):
    section("DATASET METADATA")

    extractor = MetadataExtractor()
    extractor.extract(df)


def profiling_analysis(df):
    section("DATA PROFILE")

    profiler = DataProfiler()

    if hasattr(profiler, "profile"):
        profiler.profile(df)

    elif hasattr(profiler, "analyze"):
        profiler.analyze(df)

    elif hasattr(profiler, "run"):
        profiler.run(df)

    else:
        print("DataProfiler initialized.")
        print("No compatible profiling method found.")


def quality_analysis(df):
    section("DATA QUALITY")

    engine = DataQualityEngine()

    if hasattr(engine, "analyze"):
        engine.analyze(df)

    elif hasattr(engine, "check"):
        engine.check(df)

    elif hasattr(engine, "run"):
        engine.run(df)

    else:
        print("DataQualityEngine initialized.")
        print("No compatible quality-analysis method found.")


def statistical_intelligence(df):
    section("STATISTICAL INTELLIGENCE")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:
        print("No numeric columns found.")
        return

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        mean = series.mean()
        median = series.median()
        std = series.std()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = (
            (series < lower_bound) |
            (series > upper_bound)
        ).sum()

        skewness = series.skew()

        if skewness > 1:
            distribution = "Highly Right-Skewed"

        elif skewness > 0.5:
            distribution = "Right-Skewed"

        elif skewness < -1:
            distribution = "Highly Left-Skewed"

        elif skewness < -0.5:
            distribution = "Left-Skewed"

        else:
            distribution = "Approximately Symmetric"

        print()
        print(f"Column: {column}")
        print(f"  Mean              : {mean:.2f}")
        print(f"  Median            : {median}")
        print(f"  Standard Deviation: {std:.2f}")
        print(f"  Q1                : {q1}")
        print(f"  Q3                : {q3}")
        print(f"  IQR               : {iqr}")
        print(f"  Outliers          : {outlier_count}")
        print(f"  Skewness          : {skewness:.3f}")
        print(f"  Distribution      : {distribution}")


def semantic_intelligence(df):
    section("SEMANTIC INTELLIGENCE")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    found = False

    for column in numeric_columns:

        counts = df[column].value_counts()
        threshold = len(df) * 0.05

        dominant_values = counts[
            counts >= threshold
        ]

        if dominant_values.empty:
            continue

        found = True

        print()
        print(f"Column: {column}")
        print("  Dominant / Potential Sentinel Values:")

        for value, count in dominant_values.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"    {value} -> "
                f"{count} records "
                f"({percentage:.2f}%)"
            )

    if not found:
        print("No dominant sentinel-like values detected.")


def relationship_intelligence(df):
    section("RELATIONSHIP INTELLIGENCE")

    print()
    print("---------- NUMERIC CORRELATIONS ----------")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) >= 2:

        correlation_matrix = df[
            numeric_columns
        ].corr()

        found = False

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                column_a = numeric_columns[i]
                column_b = numeric_columns[j]

                correlation = correlation_matrix.loc[
                    column_a,
                    column_b
                ]

                if pd.isna(correlation):
                    continue

                if abs(correlation) < 0.30:
                    continue

                found = True

                if correlation >= 0.70:
                    relationship = "Strong Positive"

                elif correlation >= 0.30:
                    relationship = "Moderate Positive"

                elif correlation <= -0.70:
                    relationship = "Strong Negative"

                else:
                    relationship = "Moderate Negative"

                print(
                    f"{column_a} <-> {column_b}"
                )

                print(
                    f"  Correlation : "
                    f"{correlation:.3f}"
                )

                print(
                    f"  Relationship: "
                    f"{relationship}"
                )

        if not found:
            print(
                "No meaningful numeric "
                "relationships found."
            )

    else:
        print(
            "Not enough numeric columns "
            "for correlation analysis."
        )

    print()
    print(
        "---------- CATEGORICAL "
        "TARGET RELATIONSHIPS ----------"
    )

    if "y" not in df.columns:
        print("Target column 'y' does not exist.")
        return

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:

        if column == "y":
            continue

        print()
        print(f"Column: {column}")

        try:

            target_rates = (
                df.groupby(
                    column,
                    dropna=False
                )["y"]
                .apply(
                    lambda values:
                    (values == "yes").mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            for category, rate in target_rates.head(5).items():

                print(
                    f"  {category} -> "
                    f"Target=yes: "
                    f"{rate:.2f}%"
                )

        except Exception as error:

            logger.warning(
                "Could not analyze %s: %s",
                column,
                error
            )


def anomaly_intelligence(df):
    section("ANOMALY INTELLIGENCE")

    engine = AnomalyEngine()
    engine.analyze(df)


def main():

    print()
    print("=" * 70)
    print(
        "        ENTERPRISE DATA INTELLIGENCE PLATFORM"
    )
    print("=" * 70)

    try:

        df = load_data()

        validate_data(df)

        metadata_analysis(df)

        profiling_analysis(df)

        quality_analysis(df)

        statistical_intelligence(df)

        semantic_intelligence(df)

        relationship_intelligence(df)

        anomaly_intelligence(df)

        section("PIPELINE COMPLETE")

        print()
        print("Dataset successfully processed.")
        print()
        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")
        print()
        print(
            "Enterprise Data Intelligence "
            "Pipeline completed successfully."
        )

    except FileNotFoundError:

        logger.error(
            "Dataset not found: %s",
            DATA_FILE
        )

        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print()
        print(
            f"Could not find dataset:\n"
            f"{DATA_FILE}"
        )

        sys.exit(1)

    except Exception as error:

        logger.exception(
            "Pipeline execution failed."
        )

        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print()
        print(f"Error: {error}")

        sys.exit(1)


if __name__ == "__main__":
    main()