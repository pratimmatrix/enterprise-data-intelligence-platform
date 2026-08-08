# ============================================================
# ENTERPRISE DATA INTELLIGENCE PLATFORM
# MAIN APPLICATION PIPELINE
# ============================================================

import logging
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# IMPORTS
# ============================================================

from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator

from src.profiling.profiler import DataProfiler
from src.profiling.anomalies import AnomalyEngine

from src.feature_engineering.feature_engineer import (
    FeatureEngineeringEngine
)

from src.feature_validation.feature_validator import (
    FeatureValidator
)

from src.modeling.model_trainer import ModelTrainer
from src.modeling.model_comparator import ModelComparator
from src.modeling.model_selector import ModelSelector


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = PROJECT_ROOT / "bank-full.csv"

TARGET_COLUMN = "y"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title):
    """Print a consistent section heading."""

    print()
    print("=" * 70)
    print(f"                    {title}")
    print("=" * 70)


def find_method(obj, possible_names):
    """
    Find the first available callable method from a list.
    """

    for name in possible_names:

        method = getattr(obj, name, None)

        if callable(method):
            return method

    return None


def run_dataframe_operation(
    obj,
    possible_names,
    df,
    operation_name
):
    """
    Execute a dataframe-processing method safely.

    The project has multiple independent engines. Their exact
    method names can differ, so this function checks the available
    public API before execution.
    """

    method = find_method(
        obj,
        possible_names
    )

    if method is None:

        logger.warning(
            "%s method not found. "
            "Skipping this stage.",
            operation_name
        )

        return df

    try:

        result = method(df)

        # ----------------------------------------------------
        # Most engines return a dataframe.
        # Some validation/reporting methods return None.
        # ----------------------------------------------------

        if isinstance(result, pd.DataFrame):

            return result

        if result is None:

            return df

        # Some feature engineering implementations may return
        # a tuple such as (df, metadata).

        if isinstance(result, tuple):

            for item in result:

                if isinstance(item, pd.DataFrame):

                    return item

        logger.warning(
            "%s returned %s instead of DataFrame. "
            "Keeping original dataframe.",
            operation_name,
            type(result).__name__
        )

        return df

    except TypeError:

        # Some engines expose methods that don't require df.
        # Try a zero-argument call if possible.

        try:

            result = method()

            if isinstance(result, pd.DataFrame):
                return result

            return df

        except Exception as exc:

            logger.warning(
                "%s could not be executed: %s",
                operation_name,
                exc
            )

            return df

    except Exception as exc:

        logger.warning(
            "%s failed: %s",
            operation_name,
            exc
        )

        return df


# ============================================================
# DATA INGESTION
# ============================================================

def data_ingestion():

    print_section("DATA INGESTION")

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}\n\n"
            "Make sure bank-full.csv is located in the "
            "project root."
        )

    logger.info(
        "Starting data ingestion..."
    )

    loader = DataLoader()

    # --------------------------------------------------------
    # Try common loader APIs used by the project.
    # --------------------------------------------------------

    method = find_method(
        loader,
        [
            "load_csv",
            "load_data",
            "load",
            "read_csv",
            "ingest"
        ]
    )

    if method is None:

        # Safe fallback.
        logger.warning(
            "No supported DataLoader method found. "
            "Using pandas CSV loading."
        )

        df = pd.read_csv(
            DATA_FILE,
            sep=";"
        )

    else:

        try:

            df = method(
                str(DATA_FILE)
            )

        except TypeError:

            try:
                df = method(DATA_FILE)

            except TypeError:

                df = method()

    # --------------------------------------------------------
    # Handle possible tuple/dict return values.
    # --------------------------------------------------------

    if isinstance(df, tuple):

        dataframe = None

        for item in df:

            if isinstance(item, pd.DataFrame):

                dataframe = item
                break

        df = dataframe

    elif isinstance(df, dict):

        dataframe = None

        for value in df.values():

            if isinstance(value, pd.DataFrame):

                dataframe = value
                break

        df = dataframe

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "DataLoader did not return a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Bank Marketing dataset is normally semicolon separated.
    # If a single column was produced, retry correctly.
    # --------------------------------------------------------

    if len(df.columns) == 1:

        logger.warning(
            "Dataset appears to contain one column. "
            "Retrying with ';' separator."
        )

        df = pd.read_csv(
            DATA_FILE,
            sep=";"
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

def data_validation(df):

    print_section("DATA VALIDATION")

    logger.info(
        "Starting data validation..."
    )

    validator = DataValidator()

    method = find_method(
        validator,
        [
            "validate",
            "validate_data",
            "run",
            "check"
        ]
    )

    if method is not None:

        try:
            method(df)

        except TypeError:

            try:
                method()

            except Exception as exc:

                logger.warning(
                    "Data validation warning: %s",
                    exc
                )

        except Exception as exc:

            logger.warning(
                "Data validation warning: %s",
                exc
            )

    # --------------------------------------------------------
    # Universal validation summary.
    # --------------------------------------------------------

    print()

    print(
        "Rows              :",
        len(df)
    )

    print(
        "Columns           :",
        len(df.columns)
    )

    print()

    print(
        "Duplicate Rows    :",
        df.duplicated().sum()
    )

    print()

    print(
        "Missing Values"
    )

    print(
        df.isnull().sum()
    )

    print()

    print(
        "Data Types"
    )

    print(
        df.dtypes
    )

    print()

    print(
        "Memory Usage"
    )

    print(
        f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    )

    return df


# ============================================================
# PROFILING
# ============================================================

def profiling(df):

    print_section("DATASET PROFILING")

    logger.info(
        "Starting dataset profiling..."
    )

    profiler = DataProfiler()

    method = find_method(
        profiler,
        [
            "profile",
            "profile_data",
            "run",
            "analyze",
            "generate_profile"
        ]
    )

    if method is not None:

        try:
            method(df)

        except TypeError:

            try:
                method()

            except Exception as exc:

                logger.warning(
                    "Profiling warning: %s",
                    exc
                )

        except Exception as exc:

            logger.warning(
                "Profiling warning: %s",
                exc
            )

    else:

        logger.warning(
            "No compatible DataProfiler method found."
        )

    # --------------------------------------------------------
    # Basic profiling fallback.
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print()

    print(
        "Rows             :",
        len(df)
    )

    print(
        "Columns          :",
        len(df.columns)
    )

    print(
        "Numeric Columns  :",
        len(numeric_columns)
    )

    print(
        "Text Columns     :",
        len(categorical_columns)
    )

    return df


# ============================================================
# ANOMALY INTELLIGENCE
# ============================================================

def anomaly_analysis(df):

    print_section("ANOMALY INTELLIGENCE")

    logger.info(
        "Starting anomaly detection..."
    )

    engine = AnomalyEngine()

    method = find_method(
        engine,
        [
            "detect",
            "detect_anomalies",
            "analyze",
            "run",
            "find_anomalies"
        ]
    )

    if method is not None:

        try:
            method(df)

        except TypeError:

            try:
                method()

            except Exception as exc:

                logger.warning(
                    "Anomaly detection warning: %s",
                    exc
                )

        except Exception as exc:

            logger.warning(
                "Anomaly detection warning: %s",
                exc
            )

    else:

        logger.warning(
            "No compatible anomaly method found."
        )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    print_section("FEATURE ENGINEERING")

    logger.info(
        "Starting feature engineering..."
    )

    engine = FeatureEngineeringEngine()

    method = find_method(
        engine,
        [
            "create_features",
            "engineer_features",
            "transform",
            "feature_engineering",
            "run",
            "create"
        ]
    )

    if method is None:

        raise AttributeError(
            "FeatureEngineeringEngine does not expose "
            "a supported feature-engineering method."
        )

    try:

        result = method(df)

    except TypeError:

        result = method()

    # --------------------------------------------------------
    # Extract dataframe.
    # --------------------------------------------------------

    if isinstance(result, pd.DataFrame):

        df = result

    elif isinstance(result, tuple):

        found = False

        for item in result:

            if isinstance(item, pd.DataFrame):

                df = item
                found = True
                break

        if not found:

            raise TypeError(
                "Feature engineering returned a tuple "
                "without a DataFrame."
            )

    elif result is None:

        # Some engines modify df in place.
        pass

    else:

        raise TypeError(
            "Feature engineering returned an unsupported "
            f"type: {type(result).__name__}"
        )

    print()

    print(
        "Feature engineering completed."
    )

    print(
        f"Dataset shape after feature engineering: "
        f"{df.shape}"
    )

    print()

    print("Current columns:")

    for column in df.columns:

        print(f"• {column}")

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def feature_validation(df):

    print_section("FEATURE VALIDATION")

    logger.info(
        "Starting feature validation..."
    )

    validator = FeatureValidator()

    method = find_method(
        validator,
        [
            "validate",
            "validate_features",
            "run",
            "check"
        ]
    )

    if method is None:

        logger.warning(
            "No compatible FeatureValidator method found."
        )

        return df

    try:

        method(df)

    except TypeError:

        try:
            method()

        except Exception as exc:

            logger.warning(
                "Feature validation warning: %s",
                exc
            )

    except Exception as exc:

        logger.warning(
            "Feature validation warning: %s",
            exc
        )

    return df


# ============================================================
# BASELINE MODEL
# ============================================================

def baseline_model(df):

    print_section("BASELINE MODEL TRAINING")

    logger.info(
        "Starting baseline model training..."
    )

    trainer = ModelTrainer()

    method = find_method(
        trainer,
        [
            "train",
            "run",
            "fit",
            "train_model",
            "build_model"
        ]
    )

    if method is None:

        logger.warning(
            "No compatible ModelTrainer method found."
        )

        return None

    try:

        result = method(df)

    except TypeError:

        try:
            result = method()

        except Exception as exc:

            logger.warning(
                "Baseline training warning: %s",
                exc
            )

            return None

    except Exception as exc:

        logger.warning(
            "Baseline training warning: %s",
            exc
        )

        return None

    return result


# ============================================================
# MODEL COMPARISON
# ============================================================

def model_comparison(df):

    print_section("MODEL COMPARISON")

    logger.info(
        "Starting model comparison..."
    )

    comparator = ModelComparator()

    comparison_output = comparator.run(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # ModelComparator.run() from your actual code returns:
    #
    # {
    #     "results": DataFrame,
    #     "best_model_name": str,
    #     "best_model": pipeline
    # }
    # --------------------------------------------------------

    if isinstance(
        comparison_output,
        dict
    ):

        results_df = comparison_output.get(
            "results"
        )

        best_model_name = comparison_output.get(
            "best_model_name"
        )

        best_model = comparison_output.get(
            "best_model"
        )

    elif isinstance(
        comparison_output,
        pd.DataFrame
    ):

        # Compatibility fallback if you later modify
        # ModelComparator to return only the DataFrame.

        results_df = comparison_output

        best_model_name = None
        best_model = None

    else:

        raise TypeError(
            "ModelComparator.run() must return either "
            "a pandas DataFrame or a dictionary containing "
            "'results'."
        )

    if not isinstance(
        results_df,
        pd.DataFrame
    ):

        raise TypeError(
            "Model comparison results must be "
            "a pandas DataFrame."
        )

    print()

    print(
        "Model comparison completed."
    )

    print()

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    return {
        "results": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model
    }


# ============================================================
# MODEL SELECTION
# ============================================================

def model_selection(comparison_output):

    print_section("MODEL SELECTION")

    logger.info(
        "Starting model selection..."
    )

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # ModelComparator returns a dictionary.
    # ModelSelector requires a DataFrame.
    # --------------------------------------------------------

    if isinstance(
        comparison_output,
        dict
    ):

        comparison_results = comparison_output.get(
            "results"
        )

    else:

        comparison_results = comparison_output

    if not isinstance(
        comparison_results,
        pd.DataFrame
    ):

        raise TypeError(
            "Model comparison must provide a "
            "pandas DataFrame under the 'results' key."
        )

    # --------------------------------------------------------
    # F1 is a sensible default for this project because the
    # target is imbalanced.
    #
    # However, recall is also important for identifying
    # potential customers.
    # --------------------------------------------------------

    selector = ModelSelector(
        strategy="f1"
    )

    selected_model_name = selector.select(
        comparison_results
    )

    selection_details = (
        selector.get_selection_details()
    )

    print()

    print(
        "FINAL MODEL SELECTION"
    )

    print(
        f"Selected model: {selected_model_name}"
    )

    print(
        f"Selection metric: F1"
    )

    return {
        "selected_model_name": selected_model_name,
        "selection_details": selection_details
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    df,
    comparison_output,
    selection_output
):

    print_section(
        "ENTERPRISE PIPELINE SUMMARY"
    )

    results_df = comparison_output[
        "results"
    ]

    selected_model_name = selection_output[
        "selected_model_name"
    ]

    selected_row = results_df[
        results_df["model"]
        == selected_model_name
    ].iloc[0]

    print()

    print(
        f"Dataset rows       : {len(df)}"
    )

    print(
        f"Dataset columns    : {len(df.columns)}"
    )

    print()

    print(
        "---------- SELECTED MODEL ----------"
    )

    print(
        f"Model      : {selected_model_name}"
    )

    print(
        f"Accuracy   : "
        f"{selected_row['accuracy']:.4f}"
    )

    print(
        f"Precision  : "
        f"{selected_row['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{selected_row['recall']:.4f}"
    )

    print(
        f"F1 Score   : "
        f"{selected_row['f1']:.4f}"
    )

    print(
        f"ROC-AUC    : "
        f"{selected_row['roc_auc']:.4f}"
    )

    print()

    print(
        "Enterprise Data Intelligence "
        "Pipeline completed successfully."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    try:

        print()
        print("=" * 70)
        print(
            "       ENTERPRISE DATA INTELLIGENCE PLATFORM"
        )
        print("=" * 70)

        print()

        # ----------------------------------------------------
        # 1. INGESTION
        # ----------------------------------------------------

        df = data_ingestion()

        # ----------------------------------------------------
        # 2. DATA VALIDATION
        # ----------------------------------------------------

        df = data_validation(
            df
        )

        # ----------------------------------------------------
        # 3. PROFILING
        # ----------------------------------------------------

        df = profiling(
            df
        )

        # ----------------------------------------------------
        # 4. ANOMALY INTELLIGENCE
        # ----------------------------------------------------

        df = anomaly_analysis(
            df
        )

        # ----------------------------------------------------
        # 5. FEATURE ENGINEERING
        # ----------------------------------------------------

        df = feature_engineering(
            df
        )

        # ----------------------------------------------------
        # 6. FEATURE VALIDATION
        # ----------------------------------------------------

        df = feature_validation(
            df
        )

        # ----------------------------------------------------
        # 7. BASELINE MODEL
        # ----------------------------------------------------

        baseline_result = baseline_model(
            df
        )

        # ----------------------------------------------------
        # 8. MODEL COMPARISON
        # ----------------------------------------------------

        comparison_output = model_comparison(
            df
        )

        # ----------------------------------------------------
        # 9. MODEL SELECTION
        # ----------------------------------------------------

        selection_output = model_selection(
            comparison_output
        )

        # ----------------------------------------------------
        # 10. FINAL SUMMARY
        # ----------------------------------------------------

        final_summary(
            df,
            comparison_output,
            selection_output
        )

        return 0

    except KeyboardInterrupt:

        print()

        logger.warning(
            "Pipeline interrupted by user."
        )

        return 1

    except Exception as exc:

        logger.exception(
            "Pipeline execution failed."
        )

        print()

        print(
            "ERROR:",
            str(exc)
        )

        return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )