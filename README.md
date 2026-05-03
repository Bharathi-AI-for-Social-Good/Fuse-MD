<div align="center">

# Fuse-MD

**A culturally-aware multimodal framework for misogyny meme detection in low-resource languages**

[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-black.svg)](LICENSE)
[![Dataset License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC--SA%204.0-blue.svg)](DATA_LICENSE.md)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper%20License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

Fuse-MD is a culturally-aware multimodal framework for misogyny meme detection
in low-resource languages, with experiments focused on Tamil and Malayalam.
This repository contains the source code for training, evaluation,
checkpoint-based inference, and API serving.

> Based on the paper: "Fuse-MD: A culturally-aware multimodal model for
> detecting misogyny memes" published in the *Natural Language Processing
> Journal*.

## 🧠 Overview

Fuse-MD combines:

1. A text encoder based on a language model checkpoint.
2. An image encoder based on a pretrained vision backbone from `timm`.
3. A fusion head for multimodal classification.

The current implementation supports multiple fusion strategies in code,
including `concat`, `element`, `avgpool`, and `gated`. Training and inference
are managed with Hydra configuration files under `src/configs/`.

## ✨ At a Glance

| Component | Details |
|---|---|
| Task | Multimodal misogyny meme detection |
| Languages | Tamil, Malayalam |
| Text backbone | `VishnuPJ/MalayaLLM_7B_Base` by default |
| Image backbone | `vit_base_patch16_224` |
| Fusion methods | `concat`, `element`, `avgpool`, `gated` |
| Config system | Hydra |
| API framework | FastAPI |
| Code license | MIT |
| Dataset license | CC BY-NC-SA 4.0 |

## 📁 Repository Overview

- `src/` contains the main training, inference, dataset, model, and Hydra config code.
- `api/` contains the FastAPI inference service for serving Fuse-MD predictions.
- `notebooks/` contains exploratory and unimodal experiment notebooks.
- `data/` is a local placeholder directory for your dataset copy and is not tracked.

## ⚡ Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

The dataset is not included in this repository. Place your local dataset under:

```text
data/
  <language>/
    train/
    dev/
    test/
```

By default, the configuration expects:

- `data/malayalam/train/train.csv`
- `data/malayalam/dev/dev.csv`
- `data/malayalam/test/test.csv`

Matching image files should be stored inside each split directory.

## ⚙️ Configuration

The main Hydra configuration lives in:

- `src/configs/config.yaml`
- `src/configs/dataset/default.yaml`
- `src/configs/model/default.yaml`
- `src/configs/training/default.yaml`

Default settings currently include:

- language: `malayalam`
- text model: `VishnuPJ/MalayaLLM_7B_Base`
- image model: `vit_base_patch16_224`
- fusion method: `gated`

You can override any setting from the command line through Hydra.

## 🏋️ Training

Run training from inside `src`:

```bash
cd src
python main.py
```

Example with overrides:

```bash
cd src
python main.py dataset.language=tamil model.fusion_methods=[element] training.learning_rates=[1e-5]
```

Training will:

1. Load train, dev, and test splits.
2. Tune the classification threshold on the dev set.
3. Evaluate on the test split.
4. Save checkpoints, predictions, and metrics.

Outputs are written under:

```text
trained_model/
predictions/
```

## 🔎 Inference

Run checkpoint-based inference from inside `src`:

```bash
cd src
python inference.py
```

Example with overrides:

```bash
cd src
python inference.py dataset.language=tamil dataset.inference_split=test
```

Inference uses the checkpoint path defined in
`src/configs/dataset/default.yaml` unless you override it.

## 🌐 API

Run the FastAPI service from the repository root:

```bash
uvicorn api.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Once running, you can use:

- `/docs` for the interactive Swagger UI
- `/health` for service health
- `/model-info` for loaded checkpoint metadata
- `/predict` for multimodal inference

See [api/README.md](api/README.md) for API-specific usage details.

## 🧭 Project Structure

```text
Fuse-MD/
|-- api/
|   |-- README.md
|   `-- api/
|-- data/
|   |-- .gitignore
|-- notebooks/
|-- src/
|   |-- configs/
|   |-- models/
|   |-- dataset.py
|   |-- inference.py
|   |-- main.py
|   `-- train.py
|-- DATA_LICENSE.md
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## 🗂️ Dataset

This repository does not include the dataset itself.

The dataset used with this project is released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License
(`CC BY-NC-SA 4.0`) for non-commercial academic research use. See
[DATA_LICENSE.md](DATA_LICENSE.md).

## 📜 License

Unless otherwise noted, the source code in this repository is licensed under
the MIT License. See [LICENSE](LICENSE).

The associated research paper was published open access under `CC BY 4.0`.

## 📖 Citation

If you use this repository, please cite the Fuse-MD paper:

```bibtex
@article{ponnusamy2026fusemd,
  title={Fuse-MD: A culturally-aware multimodal model for detecting misogyny memes},
  author={Ponnusamy, Rahul and Rajiakodi, Saranya and Sivagnanam, Bhuvaneswari and Kizhakkeparambil, Anshid and Sharma, Dhruv and Buitelaar, Paul and Chakravarthi, Bharathi Raja},
  journal={Natural Language Processing Journal},
  volume={14},
  pages={100197},
  year={2026},
  doi={10.1016/j.nlp.2026.100197}
}
```
