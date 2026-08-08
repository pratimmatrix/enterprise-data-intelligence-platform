import logging
import sys
import pandas as pd

from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator

from src.profiling.metadata import MetadataExtractor
from src.profiling.profiler import DataProfiler
from src.profiling.quality import DataQualityEngine
from src.profiling.anomalies import AnomalyEngine

from src.feature_engineering.feature_engineer import (
    FeatureEngineeringEngine
)

from src.feature_validation.feature_validator import (
    FeatureValidator
)

from src.modeling.model_trainer import (
    ModelTrainer
)

from src.modeling.model_comparator import (
    ModelComparator
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/raw/bank-full.csv"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# SECTION HEADER
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(f"{title:^70}")
    print("=" * 70)


# ============================================================
# DATA INGESTION
# ============================================================

def load_data():

    section("DATA INGESTION")

    logger.info(
        "Starting data ingestion..."
    )

    loader = DataLoader()

    df = loader.load(DATA_FILE)

    if df is None:

        raise RuntimeError(
            "DataLoader returned no dataframe."
        )

    print()
    print(
        "Data successfully loaded."
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(df):

    section("DATA VALIDATION")

    validator = DataValidator()

    validator.validate(df)


# ============================================================
# DATASET METADATA
# ============================================================

def metadata_analysis(df):

    section("DATASET METADATA")

    extractor = MetadataExtractor()

    extractor.extract(df)


# ============================================================
# DATA PROFILING
# ============================================================

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

        print(
            "DataProfiler initialized."
        )

        print(
            "No compatible profiling "
            "method found."
        )


# ============================================================
# DATA QUALITY
# ============================================================

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

        print(
            "DataQualityEngine initialized."
        )

        print(
            "No compatible quality-analysis "
            "method found."
        )


# ============================================================
# STATISTICAL INTELLIGENCE
# ============================================================

def statistical_intelligence(df):

    section("STATISTICAL INTELLIGENCE")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:

        print(
            "No numeric columns found."
        )

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

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outlier_count = (
            (series < lower_bound)
            |
            (series > upper_bound)
        ).sum()

        skewness = series.skew()

        if skewness > 1:

            distribution = (
                "Highly Right-Skewed"
            )

        elif skewness > 0.5:

            distribution = (
                "Right-Skewed"
            )

        elif skewness < -1:

            distribution = (
                "Highly Left-Skewed"
            )

        elif skewness < -0.5:

            distribution = (
                "Left-Skewed"
            )

        else:

            distribution = (
                "Approximately Symmetric"
            )

        print()

        print(
            f"Column: {column}"
        )

        print(
            f"  Mean              : "
            f"{mean:.2f}"
        )

        print(
            f"  Median            : "
            f"{median}"
        )

        print(
            f"  Standard Deviation: "
            f"{std:.2f}"
        )

        print(
            f"  Q1                : "
            f"{q1}"
        )

        print(
            f"  Q3                : "
            f"{q3}"
        )

        print(
            f"  IQR               : "
            f"{iqr}"
        )

        print(
            f"  Outliers          : "
            f"{outlier_count}"
        )

        print(
            f"  Skewness          : "
            f"{skewness:.3f}"
        )

        print(
            f"  Distribution      : "
            f"{distribution}"
        )


# ============================================================
# SEMANTIC INTELLIGENCE
# ============================================================

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

        print(
            f"Column: {column}"
        )

        print(
            "  Dominant / Potential "
            "Sentinel Values:"
        )

        for value, count in (
            dominant_values.items()
        ):

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"    {value} -> "
                f"{count} records "
                f"({percentage:.2f}%)"
            )

    if not found:

        print(
            "No dominant sentinel-like "
            "values detected."
        )


# ============================================================
# RELATIONSHIP INTELLIGENCE
# ============================================================

def relationship_intelligence(df):

    section("RELATIONSHIP INTELLIGENCE")

    print()

    print(
        "---------- NUMERIC CORRELATIONS ----------"
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) >= 2:

        correlation_matrix = df[
            numeric_columns
        ].corr()

        found = False

        for i in range(
            len(numeric_columns)
        ):

            for j in range(
                i + 1,
                len(numeric_columns)
            ):

                column_a = (
                    numeric_columns[i]
                )

                column_b = (
                    numeric_columns[j]
                )

                correlation = (
                    correlation_matrix.loc[
                        column_a,
                        column_b
                    ]
                )

                if pd.isna(correlation):

                    continue

                if abs(correlation) < 0.30:

                    continue

                found = True

                if correlation >= 0.70:

                    relationship = (
                        "Strong Positive"
                    )

                elif correlation >= 0.30:

                    relationship = (
                        "Moderate Positive"
                    )

                elif correlation <= -0.70:

                    relationship = (
                        "Strong Negative"
                    )

                else:

                    relationship = (
                        "Moderate Negative"
                    )

                print(
                    f"{column_a} <-> "
                    f"{column_b}"
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

        print(
            "Target column 'y' "
            "does not exist."
        )

        return

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        ).columns
    )

    for column in categorical_columns:

        if column == "y":

            continue

        print()

        print(
            f"Column: {column}"
        )

        try:

            target_rates = (
                df.groupby(
                    column,
                    dropna=False
                )["y"]
                .apply(
                    lambda values:
                    (
                        values == "yes"
                    ).mean() * 100
                )
                .sort_values(
                    ascending=False
                )
            )

            for category, rate in (
                target_rates
                .head(5)
                .items()
            ):

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


# ============================================================
# ANOMALY INTELLIGENCE
# ============================================================

def anomaly_intelligence(df):

    section("ANOMALY INTELLIGENCE")

    engine = AnomalyEngine()

    engine.analyze(df)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering_analysis(df):

    section("FEATURE ENGINEERING")

    logger.info(
        "Starting feature engineering..."
    )

    try:

        engine = FeatureEngineeringEngine()

        df = engine.create_features(df)

        print()

        print(
            "Feature engineering completed."
        )

        print(
            "Dataset shape after "
            f"feature engineering: "
            f"{df.shape}"
        )

        print()

        print(
            "Current columns:"
        )

        for column in df.columns:

            print(
                f"• {column}"
            )

        return df

    except Exception as error:

        logger.exception(
            "Feature engineering failed."
        )

        print()

        print(
            f"Feature engineering error: "
            f"{error}"
        )

        raise


# ============================================================
# FEATURE VALIDATION
# ============================================================

def feature_validation(df):

    section("FEATURE VALIDATION")

    logger.info(
        "Starting feature validation..."
    )

    validator = FeatureValidator()

    validation_result = (
        validator.validate(df)
    )

    if validation_result is False:

        raise ValueError(
            "Feature validation failed."
        )

    print()

    print(
        "Feature validation completed."
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def model_training(df):

    section("MODEL TRAINING")

    logger.info(
        "Starting model training..."
    )

    try:

        trainer = ModelTrainer()

        results = trainer.run(df)

        if results is None:

            raise RuntimeError(
                "Model training returned "
                "no results."
            )

        print()

        print(
            "Model training completed."
        )

        return results

    except Exception as error:

        logger.exception(
            "Model training failed."
        )

        print()

        print(
            f"Model training error: "
            f"{error}"
        )

        raise


# ============================================================
# MODEL COMPARISON
# ============================================================

def model_comparison(df):

    section("MODEL COMPARISON")

    logger.info(
        "Starting model comparison..."
    )

    try:

        comparator = ModelComparator()

        comparison_result = (
            comparator.run(df)
        )

        if comparison_result is None:

            raise RuntimeError(
                "Model comparison returned "
                "no result."
            )

        results = (
            comparison_result.get(
                "results"
            )
        )

        best_model_name = (
            comparison_result.get(
                "best_model_name"
            )
        )

        print()

        print(
            "---------- MODEL "
            "COMPARISON SUMMARY ----------"
        )

        if results is not None:

            print()

            print(
                results.to_string(
                    index=False,
                    float_format=lambda value:
                    f"{value:.4f}"
                )
            )

        print()

        print(
            f"Best Model: "
            f"{best_model_name}"
        )

        print()

        print(
            "Model comparison completed."
        )

        return comparison_result

    except Exception as error:

        logger.exception(
            "Model comparison failed."
        )

        print()

        print(
            f"Model comparison error: "
            f"{error}"
        )

        raise


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "        ENTERPRISE DATA "
        "INTELLIGENCE PLATFORM"
    )

    print("=" * 70)

    try:

        # ----------------------------------------------------
        # 1. DATA INGESTION
        # ----------------------------------------------------

        df = load_data()

        # ----------------------------------------------------
        # 2. DATA VALIDATION
        # ----------------------------------------------------

        validate_data(df)

        # ----------------------------------------------------
        # 3. DATASET METADATA
        # ----------------------------------------------------

        metadata_analysis(df)

        # ----------------------------------------------------
        # 4. DATA PROFILING
        # ----------------------------------------------------

        profiling_analysis(df)

        # ----------------------------------------------------
        # 5. DATA QUALITY
        # ----------------------------------------------------

        quality_analysis(df)

        # ----------------------------------------------------
        # 6. STATISTICAL INTELLIGENCE
        # ----------------------------------------------------

        statistical_intelligence(df)

        # ----------------------------------------------------
        # 7. SEMANTIC INTELLIGENCE
        # ----------------------------------------------------

        semantic_intelligence(df)

        # ----------------------------------------------------
        # 8. RELATIONSHIP INTELLIGENCE
        # ----------------------------------------------------

        relationship_intelligence(df)

        # ----------------------------------------------------
        # 9. ANOMALY INTELLIGENCE
        # ----------------------------------------------------

        anomaly_intelligence(df)

        # ----------------------------------------------------
        # 10. FEATURE ENGINEERING
        # ----------------------------------------------------

        df = feature_engineering_analysis(
            df
        )

        # ----------------------------------------------------
        # 11. FEATURE VALIDATION
        # ----------------------------------------------------

        feature_validation(df)

        # ----------------------------------------------------
        # 12. MODEL TRAINING
        # ----------------------------------------------------

        model_results = model_training(
            df
        )

        # ----------------------------------------------------
        # 13. MODEL COMPARISON
        # ----------------------------------------------------

        comparison_results = (
            model_comparison(df)
        )

        # ----------------------------------------------------
        # 14. PIPELINE COMPLETE
        # ----------------------------------------------------

        section(
            "PIPELINE COMPLETE"
        )

        print()

        print(
            "Dataset successfully processed."
        )

        print()

        print(
            f"Rows    : {df.shape[0]}"
        )

        print(
            f"Columns : {df.shape[1]}"
        )

        # ----------------------------------------------------
        # MODEL SUMMARY
        # ----------------------------------------------------

        print()

        print(
            "---------- MODEL SUMMARY ----------"
        )

        if model_results:

            print()

            print(
                "Baseline Model: "
                "Logistic Regression"
            )

            print(
                f"Accuracy : "
                f"{model_results['accuracy']:.4f}"
            )

            print(
                f"Precision: "
                f"{model_results['precision']:.4f}"
            )

            print(
                f"Recall   : "
                f"{model_results['recall']:.4f}"
            )

            print(
                f"F1 Score : "
                f"{model_results['f1']:.4f}"
            )

            print(
                f"ROC-AUC  : "
                f"{model_results['roc_auc']:.4f}"
            )

        if comparison_results:

            print()

            print(
                f"Best Model: "
                f"{comparison_results['best_model_name']}"
            )

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

        print(
            "PIPELINE FAILED"
        )

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

        print(
            "PIPELINE FAILED"
        )

        print("=" * 70)

        print()

        print(
            f"Error: {error}"
        )

        sys.exit(1)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()