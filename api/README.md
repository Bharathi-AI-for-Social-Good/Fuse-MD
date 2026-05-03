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

## Easy local inference

1. Activate your virtual environment.
2. Open `api/local_config.py` and set `CHECKPOINT_PATH` to your trained `.pth` file.
3. Start the API from the repository root:

```bash
python api/run_api.py
```

4. In a second terminal, send a prediction request:

```bash
python api/predict.py --image data/malayalam/dev/sample/148.jpg --text "sample meme transcription"
```

The helper scripts use the values in `api/local_config.py` by default, so you
only need to set the checkpoint path once for the common local workflow.

Note: the API still needs access to the underlying Hugging Face text model
files. The first startup can take a while, and if the model is not cached
locally you will need internet access to download it.

## Local config defaults

`api/local_config.py` includes:

- `CHECKPOINT_PATH`
- `HOST`
- `PORT`
- `DEVICE`
- `THRESHOLD`
- `MAX_LENGTH`

You can also override the startup settings directly from the terminal:

```bash
python api/run_api.py --checkpoint trained_model/tamil/fusion/your_checkpoint.pth --device cpu
```

## Run the API manually

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
python api/predict.py --image sample.jpg --text "sample meme transcription"
```

Manual `curl` usage still works:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "text=sample meme transcription" \
  -F "image=@sample.jpg"
```

## Notes

- The API uses the saved checkpoint metadata to rebuild the model.
- The repository does not include the dataset.
- The text backbone must be cached locally or downloadable from Hugging Face at startup.
- For single-image API inference, uploaded images are normalized per image at runtime.
