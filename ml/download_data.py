"""Download Kaggle House Prices training data into ml/data/."""

from ml.data_loader import DEFAULT_CSV_PATH, ensure_training_data


def main() -> None:
    path = ensure_training_data()
    print(f"Training data ready at {path}")


if __name__ == "__main__":
    main()
