"""FastAPI service for the text classifier."""

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import predict_text_result


PROJECT_ROOT = Path(__file__).parent
MODEL_PATHS = {
    "logistic": PROJECT_ROOT / "artifacts" / "logistic_model.joblib",
    "neural": PROJECT_ROOT / "artifacts" / "neural_model.joblib",
}

app = FastAPI(title="Text Classifier API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Input supplied by an API client."""

    text: str = Field(min_length=1, max_length=10_000)
    model: str = Field(default="logistic", pattern="^(logistic|neural)$")


@lru_cache(maxsize=2)
def _predict(model_name: str, text: str) -> dict[str, object]:
    """Run prediction using one of the project's trained model artifacts."""
    return predict_text_result(MODEL_PATHS[model_name], text)


@app.get("/check")
def health() -> dict[str, str]:
    """Report whether the service is ready to receive requests."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    """Classify user-provided text and return confidence scores."""
    model_path = MODEL_PATHS[request.model]
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact is missing: {model_path.name}. Run the training command first.",
        )
    return _predict(request.model, request.text)
