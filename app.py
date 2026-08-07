from src.ingestion.loader import DataLoader


def main():

    loader = DataLoader()

    df = loader.load_csv("data/raw/employees.csv")

    print("\nDataset Loaded Successfully!\n")

    print(df)


if __name__ == "__main__":
    main()