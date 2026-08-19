"""Fuse-MD Hugging Face ZeroGPU demo.

The Space must also contain the ``src`` and ``api`` folders and the selected
checkpoint at the repository root. The large text backbone is downloaded from
the Hugging Face Hub during startup.
"""

import os
import sys
import time
from pathlib import Path

try:
    import spaces
    GPU = spaces.GPU
except ImportError:  # pragma: no cover - local-only fallback
    def GPU(*args, **kwargs):
        def decorate(function):
            return function
        return decorate

import gradio as gr
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TEXT_MODEL = "VishnuPJ/MalayaLLM_7B_Base"
CHECKPOINT = ROOT / "custom_tamil_llamavit_fusion_element.pth"
MODEL_DIR = ROOT / ".hf_model_cache" / "VishnuPJ__MalayaLLM_7B_Base"

# The Tamil checkpoint was trained with this saved text-backbone identifier.
# Keeping it unchanged is necessary to reproduce its trained predictions.
if not MODEL_DIR.exists():
    snapshot_download(repo_id=TEXT_MODEL, local_dir=MODEL_DIR)

os.environ.setdefault("FUSEMD_CHECKPOINT", str(CHECKPOINT))
os.environ.setdefault("FUSEMD_LOCAL_MODEL_ROOT", str(ROOT / ".hf_model_cache"))
os.environ.setdefault("FUSEMD_DEVICE", "cuda")

from api.api.config import settings  # noqa: E402
from api.api.service import FuseMDService  # noqa: E402


service = FuseMDService(settings)
service.load()


@GPU(duration=120)
def predict(image_path, text, threshold):
    if image_path is None:
        return "Please upload a meme image."

    started = time.perf_counter()
    with open(image_path, "rb") as image_file:
        response = service.predict(
            text=text or "",
            image_bytes=image_file.read(),
            image_filename=Path(image_path).name,
            threshold=threshold,
        )

    elapsed = time.perf_counter() - started
    return (
        f"### Prediction: **{response['label']}**\n\n"
        f"Positive probability: **{response['probability']:.1%}**\n\n"
        f"Threshold: {response['threshold']:.2f}  |  "
        f"Language: {response['language']}  |  Fusion: {response['fusion_method']}\n\n"
        f"Runtime: **{elapsed:.1f} seconds**"
    )


with gr.Blocks(title="Fuse-MD Demo") as demo:
    gr.Markdown(
        "# Fuse-MD\n"
        "Upload a Tamil meme and enter its transcription. The model predicts "
        "whether it contains misogynistic content."
    )
    with gr.Row():
        image = gr.Image(type="filepath", label="Meme image", width=420, height=420)
        text = gr.Textbox(label="Meme transcription", lines=5)
    threshold = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Decision threshold")
    button = gr.Button("Predict", variant="primary")
    result = gr.Markdown()
    button.click(predict, inputs=[image, text, threshold], outputs=result)


demo.queue().launch()
