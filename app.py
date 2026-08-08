"""
Enterprise Data Intelligence Platform

Author: Pratim Mistry

Main application entry point.
"""

from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator

from src.profiling.metadata import MetadataExtractor
from src.profiling.profiler import DataProfiler
from src.profiling.quality import DataQualityEngine
from src.profiling.relationships import RelationshipAnalyzer


def main():

    # ==========================================
    # 1. DATA INGESTION
    # ==========================================

    loader = DataLoader()

    df = loader.load("data/raw/bank-full.csv")


    # ==========================================
    # 2. DATA VALIDATION
    # ==========================================

    validator = DataValidator()

    validator.validate(df)


    # ==========================================
    # 3. DATASET METADATA
    # ==========================================

    metadata = MetadataExtractor()

    metadata.extract(df)


    # ==========================================
    # 4. DATA PROFILING
    # ==========================================

    profiler = DataProfiler()

    profiler.profile(df)


    # ==========================================
    # 5. DATA QUALITY
    # ==========================================

    quality = DataQualityEngine()

    quality.analyze(df)


    # ==========================================
    # 6. RELATIONSHIP INTELLIGENCE
    # ==========================================

    relationship_analyzer = RelationshipAnalyzer()

    relationship_analyzer.analyze(
        df,
        target="y"
    )


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()