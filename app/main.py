from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import MODEL_PATH, STATIC_DIR, TEMPLATES_DIR
from app.routers import health_router, predict_router
from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService(MODEL_PATH)
    model_service.load()
    app.state.model_service = model_service
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a machine learning model trained on property features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=Path(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=Path(TEMPLATES_DIR))

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")
