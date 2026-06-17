from fastapi import APIRouter, HTTPException, Request

from app.schemas.prediction import HouseFeatures, PredictionResponse

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures, request: Request) -> PredictionResponse:
    model_service = request.app.state.model_service

    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        return model_service.predict(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
