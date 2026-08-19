"""Small browser demo for the running Fuse-MD FastAPI service.

Start the API first with ``python api/run_api.py``, then run:
    python gradio_demo.py
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import requests


API_URL = os.getenv("FUSEMD_API_URL", "http://127.0.0.1:8000")


def _request_prediction(image, text, threshold):
    if image is None:
        return {"error": "Please upload a meme image."}
    try:
        with open(image, "rb") as image_file:
            response = requests.post(
                f"{API_URL}/predict",
                data={"text": text or "", "threshold": str(threshold)},
                files={"image": ("meme.png", image_file, "image/png")},
                timeout=180,
            )
        response.raise_for_status()
        return {"result": response.json()}
    except requests.RequestException as exc:
        return {"error": f"API error: {exc}. Is the FastAPI service running at {API_URL}?"}


def predict(image, text, threshold):
    """Run prediction while streaming a spinner and elapsed time to the UI."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_request_prediction, image, text, threshold)
        while not future.done():
            elapsed = time.perf_counter() - started
            yield f"⏳ **Model is running…** {elapsed:.1f} seconds elapsed", ""
            time.sleep(0.25)

        response = future.result()

    elapsed = time.perf_counter() - started
    if "error" in response:
        yield f"⚠️ {response['error']}", ""
        return

    result = response["result"]
    output = (
        f"### Prediction: **{result['label']}**\n\n"
        f"Positive probability: **{result['probability']:.1%}**\n\n"
        f"Threshold: {result['threshold']:.2f}  |  "
        f"Language: {result['language']}  |  Fusion: {result['fusion_method']}\n\n"
        f"Runtime: **{elapsed:.1f} seconds**"
    )
    yield f"✅ Prediction complete in {elapsed:.1f} seconds", output

with gr.Blocks(title="Fuse-MD Demo") as demo:
    gr.Markdown(
        "# Fuse-MD\n"
        "Upload a meme and enter its transcription. The model predicts whether it "
        "contains misogynistic content."
    )
    with gr.Row():
        image = gr.Image(
            type="filepath",
            label="Meme image",
            width=420,
            height=420,
        )
        text = gr.Textbox(label="Meme transcription", lines=5, placeholder="Enter OCR/text…")
    threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Decision threshold")
    button = gr.Button("Predict", variant="primary")
    status = gr.Markdown()
    result = gr.Markdown()
    button.click(predict, inputs=[image, text, threshold], outputs=[status, result])


if __name__ == "__main__":
    demo.queue().launch()
