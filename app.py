from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator


def main():

    loader = DataLoader()

    df = loader.load("data/raw/employees.xlsx")

    validator = DataValidator()

    validator.validate(df)


if __name__ == "__main__":
    main()