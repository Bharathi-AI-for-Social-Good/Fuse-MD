from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ready: bool
    checkpoint: str
    device: str
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    ready: bool
    checkpoint: str
    device: str
    language: Optional[str] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    fusion_method: Optional[str] = None
    threshold: Optional[float] = None
    max_length: Optional[int] = None
    error: Optional[str] = None


class PredictionResponse(BaseModel):
    label: str = Field(description="Predicted class label.")
    probability: float = Field(description="Predicted probability for the positive class.")
    threshold: float = Field(description="Decision threshold used for classification.")
    predicted_positive: bool = Field(description="Whether the prediction crosses the threshold.")
    fusion_method: str
    language: str
    checkpoint: str
    image_filename: Optional[str] = None
