from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.profiling.metadata import MetadataExtractor
from src.profiling.profiler import DataProfiler


def main():

    loader = DataLoader()

    df = loader.load("data/raw/bank-full.csv")

    validator = DataValidator()
    validator.validate(df)

    metadata = MetadataExtractor()
    metadata.extract(df)

    profiler = DataProfiler()

    profile = profiler.profile(df)

    profiler.display(profile)


if __name__ == "__main__":
    main()