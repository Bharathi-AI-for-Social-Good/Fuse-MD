from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import settings
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse
from .service import FuseMDService


service = FuseMDService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Fuse-MD API",
    version="1.0.0",
    description="FastAPI service for checkpoint-based Fuse-MD multimodal inference.",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {
        "message": "Fuse-MD FastAPI service is running.",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(**service.health())


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**service.info())


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    text: str = Form(..., description="Meme transcription or OCR text."),
    image: UploadFile = File(..., description="Uploaded meme image."),
    threshold: float | None = Form(default=None, description="Optional decision threshold override."),
) -> PredictionResponse:
    try:
        image_bytes = await image.read()
        result = service.predict(
            text=text,
            image_bytes=image_bytes,
            image_filename=image.filename,
            threshold=threshold,
        )
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
