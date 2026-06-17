from pathlib import Path

import joblib
import numpy as np

from app.config import MODEL_PATH
from app.schemas.prediction import HouseFeatures, PredictionResponse


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self._model = joblib.load(self.model_path)

    def predict(self, features: HouseFeatures) -> PredictionResponse:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        input_data = np.array(
            [
                [
                    features.area_sqft,
                    features.bedrooms,
                    features.bathrooms,
                    features.age,
                    features.location_score,
                ]
            ]
        )
        predicted_price = float(self._model.predict(input_data)[0])
        return PredictionResponse(predicted_price=round(predicted_price, 2))
