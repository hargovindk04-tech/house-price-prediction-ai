"""Train and save the house price prediction model."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FEATURE_NAMES = ["area_sqft", "bedrooms", "bathrooms", "age", "location_score"]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "house_price_model.joblib"


def generate_synthetic_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    area_sqft = rng.uniform(800, 4000, n_samples)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = rng.uniform(1, 4, n_samples).round(1)
    age = rng.integers(0, 80, n_samples)
    location_score = rng.uniform(1, 10, n_samples).round(1)

    X = np.column_stack([area_sqft, bedrooms, bathrooms, age, location_score])

    # Synthetic price formula with noise
    y = (
        area_sqft * 120
        + bedrooms * 15000
        + bathrooms * 20000
        - age * 800
        + location_score * 25000
        + rng.normal(0, 25000, n_samples)
    )
    y = np.clip(y, 50000, None)

    return X, y


def train_and_save() -> Path:
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model R² score on test set: {score:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return MODEL_PATH


if __name__ == "__main__":
    train_and_save()
