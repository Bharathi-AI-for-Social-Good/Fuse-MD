import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from .config import Settings


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model, load_tokenizer_and_llama, load_trainable_checkpoint, resolve_dtype


logger = logging.getLogger(__name__)


class FuseMDService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self.resolve_device(settings.device)
        self.model = None
        self.tokenizer = None
        self.threshold: Optional[float] = settings.default_threshold
        self.max_length: Optional[int] = settings.max_length
        self.metadata: Dict[str, Any] = {}
        self.load_error: Optional[str] = None

    @staticmethod
    def resolve_device(requested_device: str) -> torch.device:
        if requested_device == "cpu":
            return torch.device("cpu")
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for the API, but no CUDA device is available.")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.load_error is None

    def load(self) -> None:
        if self.ready:
            return

        checkpoint_path = self.settings.checkpoint_path
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_cfg = checkpoint.get("config", {})
            checkpoint_dataset_cfg = self.extract_dataset_config(checkpoint_cfg)
            checkpoint_model_cfg = self.extract_model_config(checkpoint_cfg)
            checkpoint_training_cfg = self.extract_training_config(checkpoint_cfg)

            text_model_name = str(
                checkpoint_model_cfg.get("text_model", "VishnuPJ/MalayaLLM_7B_Base")
            )
            image_model_name = str(checkpoint_model_cfg.get("image_model", "vit_base_patch16_224"))
            embedding_dim = int(checkpoint_model_cfg.get("embedding_dim", 128))
            fusion_method = str(checkpoint.get("fusion_method", checkpoint_model_cfg.get("fusion_methods", ["gated"])[0]))
            language = str(checkpoint_dataset_cfg.get("language", "unknown"))
            dtype_name = str(checkpoint_model_cfg.get("dtype", "float16"))
            use_8bit = bool(checkpoint_model_cfg.get("use_8bit", False) or self.settings.use_8bit)
            max_length = int(checkpoint_training_cfg.get("max_length", self.max_length or 75))

            threshold = self.settings.default_threshold
            if threshold is None:
                threshold = float(checkpoint.get("best_threshold", 0.5))

            dtype = resolve_dtype(dtype_name)
            tokenizer, llama = load_tokenizer_and_llama(
                text_model_name,
                use_8bit=bool(use_8bit and self.device.type == "cuda"),
                dtype=dtype,
                device=self.device,
            )
            model = build_model(
                llama=llama,
                image_model_name=image_model_name,
                fusion_method=fusion_method,
                embedding_dim=embedding_dim,
            ).to(self.device)
            load_trainable_checkpoint(model, checkpoint)
            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self.threshold = threshold
            self.max_length = max_length
            self.metadata = {
                "checkpoint": str(checkpoint_path),
                "device": str(self.device),
                "language": language,
                "text_model": text_model_name,
                "image_model": image_model_name,
                "fusion_method": fusion_method,
                "threshold": threshold,
                "max_length": max_length,
            }
            self.load_error = None
            logger.info("Loaded Fuse-MD checkpoint from %s", checkpoint_path)
        except Exception as exc:  # pragma: no cover - defensive path for runtime startup failures
            self.model = None
            self.tokenizer = None
            self.metadata = {
                "checkpoint": str(checkpoint_path),
                "device": str(self.device),
            }
            self.load_error = str(exc)
            logger.exception("Failed to load Fuse-MD API model")

    @staticmethod
    def extract_dataset_config(checkpoint_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(checkpoint_cfg.get("dataset"), dict):
            return checkpoint_cfg["dataset"]
        return checkpoint_cfg

    @staticmethod
    def extract_model_config(checkpoint_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(checkpoint_cfg.get("model"), dict):
            return checkpoint_cfg["model"]
        return checkpoint_cfg

    @staticmethod
    def extract_training_config(checkpoint_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(checkpoint_cfg.get("training"), dict):
            return checkpoint_cfg["training"]
        return checkpoint_cfg

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.ready else "error",
            "ready": self.ready,
            "checkpoint": str(self.settings.checkpoint_path),
            "device": str(self.device),
            "error": self.load_error,
        }

    def info(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "checkpoint": str(self.settings.checkpoint_path),
            "device": str(self.device),
            "language": self.metadata.get("language"),
            "text_model": self.metadata.get("text_model"),
            "image_model": self.metadata.get("image_model"),
            "fusion_method": self.metadata.get("fusion_method"),
            "threshold": self.metadata.get("threshold"),
            "max_length": self.metadata.get("max_length"),
            "error": self.load_error,
        }

    def predict(self, text: str, image_bytes: bytes, image_filename: Optional[str], threshold: Optional[float]) -> Dict[str, Any]:
        if not self.ready:
            self.load()
        if not self.ready:
            raise RuntimeError(self.load_error or "Model is not available.")

        resolved_threshold = float(self.threshold if threshold is None else threshold)

        tokens = self.tokenizer(
            [text],
            truncation=True,
            max_length=int(self.max_length or 75),
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device=self.device, dtype=torch.long)
        attention_mask = tokens.attention_mask.to(device=self.device, dtype=torch.long)
        image_tensor = self.preprocess_image(image_bytes).to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, image_tensor)
            probability = float(torch.sigmoid(logits).reshape(-1)[0].item())

        predicted_positive = probability >= resolved_threshold
        return {
            "label": "misogyny" if predicted_positive else "not-misogyny",
            "probability": probability,
            "threshold": resolved_threshold,
            "predicted_positive": predicted_positive,
            "fusion_method": str(self.metadata["fusion_method"]),
            "language": str(self.metadata["language"]),
            "checkpoint": str(self.settings.checkpoint_path),
            "image_filename": image_filename,
        }

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> torch.Tensor:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the uploaded image.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # The training pipeline normalizes each split using observed image statistics.
        # For single-sample API inference, use per-image channel normalization.
        mean = image_tensor.mean(dim=(1, 2), keepdim=True)
        std = image_tensor.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        image_tensor = (image_tensor - mean) / std

        return image_tensor.unsqueeze(0)
