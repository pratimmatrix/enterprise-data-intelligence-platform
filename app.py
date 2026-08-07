from src.ingestion.loader import DataLoader


def main():

    loader = DataLoader()

    df = loader.load("data/raw/employees.xlsx")

    print(df.head())


if __name__ == "__main__":
    main()