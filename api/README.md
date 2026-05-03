# Fuse-MD API

This folder contains the FastAPI-based inference service for Fuse-MD.

## What it does

The API loads a saved Fuse-MD checkpoint and exposes HTTP endpoints for:

- health checks
- model metadata
- multimodal prediction from text and an uploaded image

## Endpoints

- `GET /`
- `GET /health`
- `GET /model-info`
- `POST /predict`

## Run the API

From the repository root:

```bash
uvicorn api.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Optional environment variables

- `FUSEMD_CHECKPOINT`
- `FUSEMD_HOST`
- `FUSEMD_PORT`
- `FUSEMD_THRESHOLD`
- `FUSEMD_MAX_LENGTH`
- `FUSEMD_USE_8BIT`
- `FUSEMD_DEVICE`

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "text=sample meme transcription" \
  -F "image=@sample.jpg"
```

## Notes

- The API uses the saved checkpoint metadata to rebuild the model.
- The repository does not include the dataset.
- For single-image API inference, uploaded images are normalized per image at runtime.
