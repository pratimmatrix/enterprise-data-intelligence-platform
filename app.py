from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.profiling.metadata import MetadataExtractor
from src.profiling.profiler import DataProfiler
from src.profiling.quality import DataQualityEngine


def main():

    # ==============================
    # 1. LOAD DATA
    # ==============================

    loader = DataLoader()

    df = loader.load("data/raw/bank-full.csv")


    # ==============================
    # 2. DATA VALIDATION
    # ==============================

    validator = DataValidator()

    validator.validate(df)


    # ==============================
    # 3. DATASET METADATA
    # ==============================

    metadata = MetadataExtractor()

    metadata.extract(df)


    # ==============================
    # 4. DATA PROFILING
    # ==============================

    profiler = DataProfiler()

    profile = profiler.profile(df)

    profiler.display(profile)


    # ==============================
    # 5. DATA QUALITY ANALYSIS
    # ==============================

    quality_engine = DataQualityEngine()

    quality_results = quality_engine.analyze(df)

    quality_engine.display(quality_results)


if __name__ == "__main__":
    main()