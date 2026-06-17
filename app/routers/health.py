from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(request: Request) -> dict:
    model_service = request.app.state.model_service
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded,
    }
