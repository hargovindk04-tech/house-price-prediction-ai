from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    area_sqft: float = Field(..., gt=0, description="Living area in square feet")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=1, le=10, description="Number of bathrooms")
    age: int = Field(..., ge=0, le=150, description="Property age in years")
    location_score: float = Field(
        ..., ge=1, le=10, description="Location desirability score (1-10)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "area_sqft": 1800,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "age": 15,
                    "location_score": 7.5,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted house price in USD")
    currency: str = Field(default="USD", description="Price currency")
