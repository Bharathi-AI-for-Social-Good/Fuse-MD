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
2. Download the text backbone once:

```bash
python api/setup_local_model.py --model VishnuPJ/MalayaLLM_7B_Base
```

3. Open `api/local_config.py` and set `CHECKPOINT_PATH` to your trained `.pth` file.
4. Start the API from the repository root:

```bash
python api/run_api.py
```

5. In a second terminal, send a prediction request:

```bash
python api/predict.py --image data/malayalam/dev/sample/148.jpg --text "sample meme transcription"
```

The helper scripts use the values in `api/local_config.py` by default, so you
only need to set the checkpoint path once for the common local workflow.

The setup command requires internet only the first time. After that, runtime
loads the text backbone from `local_models/` in offline mode. If you use a
Tamil checkpoint, run
`python api/setup_local_model.py --model abhinand/tamil-llama-7b-base-v0.1`
once too.

## Local config defaults

`api/local_config.py` includes:

- `CHECKPOINT_PATH`
- `LOCAL_MODEL_ROOT`
- `HOST`
- `PORT`
- `DEVICE`
- `THRESHOLD`
- `MAX_LENGTH`

You can also override the startup settings directly from the terminal:

```bash
python api/run_api.py --checkpoint trained_model/tamil/fusion/your_checkpoint.pth --device cpu
```

The model setup command also supports checkpoint-based inference of the model id:

```bash
python api/setup_local_model.py --checkpoint trained_model/malayalam/fusion/your_checkpoint.pth
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
- `FUSEMD_LOCAL_MODEL_ROOT`

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
- The text backbone must be downloaded once into `local_models/` before offline runtime.
- For single-image API inference, uploaded images are normalized per image at runtime.
