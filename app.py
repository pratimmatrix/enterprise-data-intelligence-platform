from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.profiling.metadata import MetadataExtractor


def main():

    loader = DataLoader()

    df = loader.load("data/raw/employees.xlsx")

    validator = DataValidator()
    validator.validate(df)

    metadata = MetadataExtractor()
    metadata.extract(df)


if __name__ == "__main__":
    main()