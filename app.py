from src.ingestion.loader import DataLoader


def main():

    loader = DataLoader()

    df = loader.load_csv("data/raw/employees.csv")

    print("\n========== DATASET INFORMATION ==========\n")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst Five Rows:")
    print(df.head())


if __name__ == "__main__":
    main()