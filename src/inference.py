import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dataset import MemeDataset
from models import build_model, load_tokenizer_and_llama, load_trainable_checkpoint, resolve_dtype
from train import collect_probabilities


logger = logging.getLogger(__name__)


def template_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_template_path(raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (template_root() / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_probs(probs: np.ndarray) -> np.ndarray:
    return probs.reshape(-1)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(cfg.training.seed)

    checkpoint_path = resolve_template_path(cfg.dataset.finetuned_model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.model.dtype)
    use_8bit = bool(cfg.model.use_8bit and device.type == "cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = checkpoint.get("config", {})
    checkpoint_dataset_cfg = checkpoint_cfg.get("dataset", {})
    checkpoint_model_cfg = checkpoint_cfg.get("model", {})

    fusion_method = str(checkpoint.get("fusion_method", cfg.model.fusion_methods[0]))
    threshold = cfg.training.threshold
    if threshold is None:
        threshold = float(checkpoint.get("best_threshold", 0.5))

    text_model_name = str(checkpoint_model_cfg.get("text_model", cfg.model.text_model))
    image_model_name = str(checkpoint_model_cfg.get("image_model", cfg.model.image_model))
    embedding_dim = int(checkpoint_model_cfg.get("embedding_dim", cfg.model.embedding_dim))
    language = str(checkpoint_dataset_cfg.get("language", cfg.dataset.language))
    image_id_col = str(checkpoint_dataset_cfg.get("image_id_col", cfg.dataset.image_id_col))
    text_col = str(checkpoint_dataset_cfg.get("text_col", cfg.dataset.text_col))
    label_col = str(checkpoint_dataset_cfg.get("label_col", cfg.dataset.label_col))

    tokenizer, llama = load_tokenizer_and_llama(text_model_name, use_8bit, dtype, device)
    dataset = MemeDataset(
        split=cfg.dataset.inference_split,
        data_root=resolve_template_path(cfg.dataset.data_root),
        language=language,
        tokenizer=tokenizer,
        image_id_col=image_id_col,
        text_col=text_col,
        label_col=label_col,
        max_length=cfg.training.max_length,
        oversample_positive_train=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
    )

    model = build_model(llama, image_model_name, fusion_method, embedding_dim).to(device)
    load_trainable_checkpoint(model, checkpoint)
    model.eval()

    probs, labels, image_ids = collect_probabilities(model, loader, device)
    probs = flatten_probs(probs)
    predictions = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    image_ids = image_ids.astype(int)

    preds_path = resolve_template_path(cfg.dataset.save_preds_path)
    metrics_path = resolve_template_path(cfg.dataset.save_metrics_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        preds_path,
        np.column_stack([image_ids, probs, predictions, labels]),
        delimiter=",",
        header="image_id,probability,prediction,true_label",
        comments="",
    )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": cfg.dataset.inference_split,
        "fusion_method": fusion_method,
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, zero_division=0),
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    logger.info("Saved predictions: %s", preds_path)
    logger.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
