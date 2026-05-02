## Fuse-MD Project

This project uses the Fuse-MD workflow directly from the repository root.

## License

Unless otherwise noted, the source code in this repository is licensed under
the MIT License. See [LICENSE](LICENSE).

The associated research paper, "Fuse-MD: A culturally-aware multimodal model
for detecting misogyny memes," was published open access under the
Creative Commons Attribution 4.0 International License (CC BY 4.0). That
article license applies to the paper and any paper content reused from it,
not automatically to all repository contents.

The dataset used with this project is released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License
(CC BY-NC-SA 4.0) and may only be used for non-commercial, academic research
purposes. The repository does not include the dataset itself; the `data/`
directory is intended as a local placeholder for your own copy. See
[DATA_LICENSE.md](DATA_LICENSE.md) for repository guidance.

### Entry points

- `src/main.py` handles training, validation, threshold tuning, and test evaluation.
- `src/inference.py` loads a saved checkpoint and writes predictions and metrics.

### Configuration

Hydra configs live under `src/configs/`:

- `dataset/default.yaml`
- `model/default.yaml`
- `training/default.yaml`

The default dataset root is `data`, so training and inference read from your
local `data/` directory.

### Typical usage

From inside `src`:

```bash
python main.py
python inference.py
```
