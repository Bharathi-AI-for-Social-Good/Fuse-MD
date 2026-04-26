# Fuse-MD

Fuse-MD is a simple multimodal misogyny detection project. It uses both meme text and meme images for binary classification.

The main runnable file is `fuse_md.py`.

## Project Structure

```text
Fuse-MD/
  dataset/
    malayalam/
      train/
      dev/
      test/
    tamil/
      train/
      dev/
      test/
  experiments/
    older experiment scripts and notebooks
  fuse_md.py
  README.md
```

## Data Format

Each language folder should contain `train`, `dev`, and `test` folders.

Each split folder should contain:

```text
train.csv
1.jpg
2.jpg
...
```

For `dev`, the CSV should be named `dev.csv`. For `test`, it should be named `test.csv`.

The default CSV columns are:

```text
image_id, transcriptions, original_labels
```

Supported labels:

```text
misogyny
not-misogyny
```

The image file name should match the `image_id` value. For example, `125.jpg` should have `image_id` as `125` in the CSV.

## Install Requirements

Install the Python packages with:

```bash
pip install -r requirements.txt
```

If you use a GPU, install the PyTorch build that matches your CUDA version from the official PyTorch install instructions, then run the command above.

## Hugging Face Access

The script downloads the text model from Hugging Face. You can run without logging in, but downloads may be slower or rate limited.

To avoid that warning, create a Hugging Face access token and set it before running training:

```bash
set HF_TOKEN=your_token_here
```

On PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

## What the Model Uses

- Text encoder: LLaMA-based model
- Image encoder: ViT from `timm`
- Fusion methods: `concat`, `element`, `avgpool`, `gated`
- Task: binary classification

## Run Training

Malayalam:

```bash
python fuse_md.py --language malayalam
```

By default, Malayalam uses the `gated` fusion method.

Tamil:

```bash
python fuse_md.py --language tamil
```

By default, Tamil uses the `element` fusion method.

Run only one fusion method:

```bash
python fuse_md.py --language malayalam --fusion-methods concat
```

Use a different batch size or number of epochs:

```bash
python fuse_md.py --language malayalam --batch-size 8 --epochs 5
```

## Output Files

After training, files are saved automatically.

Model checkpoints:

```text
trained_model/<language>/fusion/
```

Predictions and metrics:

```text
predictions/<language>/fusion/
```

Saved outputs include:

- `.pth` model checkpoint
- `.csv` predictions
- `.json` metrics
- `.txt` classification report
- `.png` confusion matrix

## Run Inference

Use `infer.py` with a saved checkpoint:

```bash
python infer.py --checkpoint trained_model/malayalam/fusion/model.pth --language malayalam --split test
```

You can also choose the output CSV path:

```bash
python infer.py --checkpoint trained_model/tamil/fusion/model.pth --language tamil --split test --output tamil_predictions.csv
```

If `--output` is not given, predictions are saved to:

```text
predictions/<language>/inference/
```

## Notes

- The dataset folder should stay in the project root.
- `fuse_md.py` should stay in the project root.
- Older scripts and notebooks are stored in `experiments/`.
- Training may require a GPU because the text model is large.
