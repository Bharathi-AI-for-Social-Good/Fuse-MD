# Project structure

The `src` folder hosts the Fuse-MD implementation.

```
configs/      Hydra configuration for dataset, model, and training
models/       Fuse-MD model components
dataset.py    Multimodal meme dataset loader
train.py      Training and evaluation helpers
main.py       Training entrypoint
inference.py  Checkpoint inference entrypoint
```

The default data location is `../data` relative to `src/`, configured as `data` from the project root in `configs/dataset/default.yaml`.
