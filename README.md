## Fuse-MD Project

This project uses the Fuse-MD workflow directly from the repository root.

### Entry points

- `src/main.py` handles training, validation, threshold tuning, and test evaluation.
- `src/inference.py` loads a saved checkpoint and writes predictions and metrics.

### Configuration

Hydra configs live under `src/configs/`:

- `dataset/default.yaml`
- `model/default.yaml`
- `training/default.yaml`

The default dataset root is `data`, so training and inference read directly from `data/`.

### Typical usage

From inside `src`:

```bash
python main.py
python inference.py
```
