"""Load and prepare house price training data from CSV files."""

from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = ["area_sqft", "bedrooms", "bathrooms", "age", "location_score"]
TARGET_COLUMN = "SalePrice"

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CSV_PATH = DATA_DIR / "house_prices.csv"
FALLBACK_CSV_PATH = DATA_DIR / "train.csv"
KAGGLE_COMPETITION = "house-prices-advanced-regression-techniques"

LOCATION_SCORES = {
    "Rural": 4.0,
    "Suburban": 6.0,
    "Urban": 7.5,
    "Downtown": 9.0,
}


def resolve_csv_path(csv_path: Path | None = None) -> Path:
    """Pick the training CSV path from argument, env var, or defaults."""
    if csv_path is not None:
        return csv_path

    env_path = os.getenv("TRAINING_DATA_PATH")
    if env_path:
        return Path(env_path)

    if DEFAULT_CSV_PATH.exists():
        return DEFAULT_CSV_PATH
    if FALLBACK_CSV_PATH.exists():
        return FALLBACK_CSV_PATH
    return DEFAULT_CSV_PATH


def ensure_training_data(csv_path: Path | None = None) -> Path:
    """Return a local CSV path, downloading from Kaggle when missing."""
    path = resolve_csv_path(csv_path)
    if path.exists():
        return path

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            f"Training data not found at {path}. Place a CSV in {DATA_DIR} "
            f"(e.g. house_prices.csv), or install kagglehub and authenticate."
        ) from exc

    print(f"Downloading {KAGGLE_COMPETITION} from Kaggle...")
    try:
        download_dir = kagglehub.competition_download(
            KAGGLE_COMPETITION,
           
        )
    except Exception as exc:
        exc_name = exc.__class__.__name__
        if exc_name == "UnauthenticatedError":
            raise FileNotFoundError(
                f"Training data not found at {path}. Place your CSV in {DATA_DIR}, "
                f"or authenticate with Kaggle and accept the competition rules at "
                f"https://www.kaggle.com/c/{KAGGLE_COMPETITION}, then run: "
                f"python -m ml.download_data"
            ) from exc
        raise

    downloaded_csv = Path(download_dir) / "train.csv"
    if not downloaded_csv.exists():
        raise FileNotFoundError(
            f"Download completed but train.csv was not found in {download_dir}"
        )

    if downloaded_csv.resolve() != path.resolve():
        shutil.copy2(downloaded_csv, path)

    return path


def _detect_format(columns: list[str]) -> str:
    column_set = set(columns)
    if "Price" in column_set and "Area" in column_set:
        return "house_price"
    if "SalePrice" in column_set and "GrLivArea" in column_set:
        return "ames"
    raise ValueError(
        "Unrecognized CSV format. Expected either the House Price Prediction Dataset "
        "(Area, Bedrooms, Bathrooms, YearBuilt, Location, Price) or Kaggle Ames "
        "(GrLivArea, BedroomAbvGr, SalePrice, ...)."
    )


def _bathroom_count(row: pd.Series) -> float:
    full = row.get("FullBath", 0) + row.get("BsmtFullBath", 0)
    half = row.get("HalfBath", 0) + row.get("BsmtHalfBath", 0)
    return float(full + 0.5 * half)


def _load_house_price_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required_columns = [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "YearBuilt",
        "Location",
        "Price",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    reference_year = int(df["YearBuilt"].max()) if not df.empty else date.today().year
    unknown_locations = sorted(set(df["Location"]) - set(LOCATION_SCORES))
    if unknown_locations:
        raise ValueError(
            f"Unknown Location values: {', '.join(unknown_locations)}. "
            f"Supported values: {', '.join(LOCATION_SCORES)}"
        )

    features = pd.DataFrame(
        {
            "area_sqft": df["Area"],
            "bedrooms": df["Bedrooms"],
            "bathrooms": df["Bathrooms"],
            "age": reference_year - df["YearBuilt"],
            "location_score": df["Location"].map(LOCATION_SCORES),
        }
    )
    target = df["Price"]
    return _finalize_dataset(features, target)


def _load_ames_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required_columns = [
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "YearBuilt",
        "YrSold",
        "OverallQual",
        TARGET_COLUMN,
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    features = pd.DataFrame(
        {
            "area_sqft": df["GrLivArea"],
            "bedrooms": df["BedroomAbvGr"],
            "bathrooms": df.apply(_bathroom_count, axis=1),
            "age": df["YrSold"] - df["YearBuilt"],
            "location_score": df["OverallQual"],
        }
    )
    target = df[TARGET_COLUMN]
    return _finalize_dataset(features, target)


def _finalize_dataset(
    features: pd.DataFrame, target: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    dataset = pd.concat([features, target.rename(TARGET_COLUMN)], axis=1).dropna()
    dataset = dataset[
        (dataset["area_sqft"] > 0)
        & (dataset["bedrooms"] >= 1)
        & (dataset["bathrooms"] >= 1)
        & (dataset["age"] >= 0)
        & (dataset["location_score"] >= 1)
        & (dataset[TARGET_COLUMN] > 0)
    ]

    X = dataset[FEATURE_NAMES].to_numpy(dtype=float)
    y = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def load_training_data(csv_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from CSV, auto-detecting supported formats."""
    path = csv_path or ensure_training_data()
    df = pd.read_csv(path)
    data_format = _detect_format(list(df.columns))

    if data_format == "house_price":
        return _load_house_price_dataset(df)
    return _load_ames_dataset(df)


def load_kaggle_data(csv_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for load_training_data."""
    return load_training_data(csv_path)
