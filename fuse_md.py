"""
Fuse-MD updated implementation template.

This single-file implementation keeps the methodology of Fuse-MD while
refactoring the experimental script into a cleaner, configurable pipeline.

Core methodological choices included:
1. LLaMA text encoder with optional 8-bit quantization.
2. ViT image encoder from timm.
3. Frozen pretrained bases, trainable projection heads and fusion module.
4. Four fusion methods: concat, element, avgpool, gated.
5. BCEWithLogitsLoss for binary classification.
6. ReduceLROnPlateau and early stopping.
7. Threshold optimization on the development set using macro-F1.
8. Trainable-only checkpoint saving to avoid quantized model serialization issues.

Expected data layout:
dataset/
  tamil/
    train/train.csv
    train/<image_id>.<ext>
    dev/dev.csv
    dev/<image_id>.<ext>
    test/test.csv
    test/<image_id>.<ext>
  malayalam/
    train/train.csv
    dev/dev.csv
    test/test.csv

Expected CSV columns by default:
  image_id, transcriptions, original_labels

Labels may be numeric 0/1 values or the dataset strings:
  misogyny, not-misogyny

Example:
python fuse_md.py \
  --data-root dataset \
  --language malayalam \
  --text-model VishnuPJ/MalayaLLM_7B_Base \
  --fusion-methods concat element avgpool gated
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class Config:
    data_root: str = "dataset"
    language: str = "malayalam"
    text_model: str = "VishnuPJ/MalayaLLM_7B_Base"
    image_model: str = "vit_base_patch16_224"
    output_root: str = "."
    image_id_col: str = "image_id"
    text_col: str = "transcriptions"
    label_col: str = "original_labels"
    max_length: int = 75
    batch_size: int = 16
    epochs: int = 10
    learning_rates: Tuple[float, ...] = (1e-5,)
    fusion_methods: Tuple[str, ...] = ("concat", "element", "avgpool", "gated")
    early_stopping_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_threshold: float = 1e-3
    optimizer_eps: float = 1e-4
    use_8bit: bool = True
    dtype: str = "float16"
    seed: int = 42
    num_workers: int = 0
    oversample_positive_train: int = 3
    threshold_start: float = 0.1
    threshold_stop: float = 0.9
    threshold_step: float = 0.1
    save_confusion_matrix_png: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fuse-MD updated implementation")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--language", choices=["tamil", "malayalam"], default="malayalam")
    parser.add_argument("--text-model", default="VishnuPJ/MalayaLLM_7B_Base")
    parser.add_argument("--image-model", default="vit_base_patch16_224")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--image-id-col", default="image_id")
    parser.add_argument("--text-col", default="transcriptions")
    parser.add_argument("--label-col", default="original_labels")
    parser.add_argument("--max-length", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5])
    parser.add_argument(
        "--fusion-methods",
        nargs="+",
        choices=["concat", "element", "avgpool", "gated"],
        default=["concat", "element", "avgpool", "gated"],
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--no-8bit", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--oversample-positive-train", type=int, default=3)
    args = parser.parse_args()

    return Config(
        data_root=args.data_root,
        language=args.language,
        text_model=args.text_model,
        image_model=args.image_model,
        output_root=args.output_root,
        image_id_col=args.image_id_col,
        text_col=args.text_col,
        label_col=args.label_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rates=tuple(args.learning_rates),
        fusion_methods=tuple(args.fusion_methods),
        early_stopping_patience=args.early_stopping_patience,
        use_8bit=not args.no_8bit,
        dtype=args.dtype,
        seed=args.seed,
        num_workers=args.num_workers,
        oversample_positive_train=args.oversample_positive_train,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def normalize_label(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"misogyny", "misogynous", "1", "true"}:
            return 1.0
        if normalized in {"not-misogyny", "not misogyny", "non-misogyny", "0", "false"}:
            return 0.0
        raise ValueError(f"Unsupported label value: {value!r}")

    return float(value)


class MemeDataset(Dataset):
    def __init__(
        self,
        split: str,
        language_path: Path,
        tokenizer: AutoTokenizer,
        config: Config,
    ) -> None:
        self.split = split
        self.split_path = language_path / split
        self.tokenizer = tokenizer
        self.config = config

        csv_path = self.split_path / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")

        df = pd.read_csv(csv_path).dropna(axis=0).copy()
        required_cols = {config.image_id_col, config.text_col, config.label_col}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing_cols)}")

        df[config.image_id_col] = df[config.image_id_col].astype(int)
        df[config.label_col] = df[config.label_col].map(normalize_label)
        id_to_row = {int(row[config.image_id_col]): row for _, row in df.iterrows()}

        raw_images: List[torch.Tensor] = []
        raw_records: List[Tuple[int, str, float]] = []
        for image_file in sorted(self.split_path.iterdir()):
            if image_file.suffix.lower() == ".csv":
                continue
            if image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            try:
                image_id = int(image_file.stem)
            except ValueError:
                continue

            if image_id not in id_to_row:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                raise ValueError(f"Could not read image: {image_file}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            raw_images.append(image_tensor)

            row = id_to_row[image_id]
            raw_records.append(
                (
                    image_id,
                    str(row[config.text_col]),
                    float(row[config.label_col]),
                )
            )

        if not raw_records:
            raise ValueError(f"No valid examples found in {self.split_path}")

        image_stack = torch.stack(raw_images, dim=0)
        mean = image_stack.mean(dim=(0, 2, 3), keepdim=True)
        std = image_stack.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        image_stack = (image_stack - mean) / std

        self.images: List[torch.Tensor] = []
        self.labels: List[float] = []
        self.image_ids: List[int] = []
        texts: List[str] = []

        for idx, (image_id, text, label) in enumerate(raw_records):
            repeat = 1
            if split == "train" and int(label) == 1 and config.oversample_positive_train > 0:
                repeat += config.oversample_positive_train

            for _ in range(repeat):
                self.images.append(image_stack[idx])
                self.labels.append(label)
                self.image_ids.append(image_id)
                texts.append(text)

        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = tokens.input_ids
        self.attention_mask = tokens.attention_mask

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "image_id": torch.tensor(self.image_ids[index], dtype=torch.long),
        }


class ImageEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding = timm.create_model(model_name, pretrained=True, num_classes=embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.embedding(images)


class LlamaTextClassifier(nn.Module):
    def __init__(self, llama: AutoModelForCausalLM, hidden_size: int = 4096) -> None:
        super().__init__()
        self.text_base = llama.model
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_size, 512, bias=False),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1, bias=False),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.clf_head(pooled)


class TextEmbeddingModel(nn.Module):
    def __init__(self, text_classifier: LlamaTextClassifier, embedding_dim: int = 128) -> None:
        super().__init__()
        text_classifier.clf_head[4] = nn.Linear(512, embedding_dim, bias=False)
        self.embedding = text_classifier

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids, attention_mask)


class FusionHead(nn.Module):
    def __init__(self, fusion: str = "concat", embedding_dim: int = 128) -> None:
        super().__init__()
        self.fusion = fusion
        self.embedding_dim = embedding_dim

        if fusion == "concat":
            fused_dim = embedding_dim * 2
        elif fusion == "element":
            fused_dim = embedding_dim
        elif fusion == "avgpool":
            self.w_image = nn.Linear(embedding_dim, 64, bias=False)
            self.w_text = nn.Linear(embedding_dim, 64, bias=False)
            self.pool = nn.AvgPool1d(kernel_size=4)
            fused_dim = 16
        elif fusion == "gated":
            self.u_text = nn.Linear(embedding_dim, 64, bias=False)
            self.v_image = nn.Linear(embedding_dim, 64, bias=False)
            self.w_fused = nn.Linear(64, 8, bias=False)
            self.sigmoid = nn.Sigmoid()
            fused_dim = 8
        else:
            raise ValueError(f"Unsupported fusion method: {fusion}")

        self.initial = nn.Linear(fused_dim, fused_dim // 2, bias=True)
        self.final = nn.Linear(fused_dim // 2, 1, bias=True)

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            fused = torch.cat([text_features, image_features], dim=-1)
        elif self.fusion == "element":
            fused = text_features * image_features
        elif self.fusion == "avgpool":
            text_encoded = self.w_text(text_features)
            image_encoded = self.w_image(image_features)
            fused = self.pool((text_encoded * image_encoded).unsqueeze(1)).squeeze(1)
        elif self.fusion == "gated":
            gate = self.sigmoid(self.u_text(text_features))
            image_encoded = self.v_image(image_features)
            fused = self.w_fused(gate * image_encoded)
        else:
            raise RuntimeError("Invalid fusion state")

        hidden = self.initial(fused)
        return self.final(hidden).squeeze(-1)


class FuseMD(nn.Module):
    def __init__(
        self,
        text_encoder: TextEmbeddingModel,
        image_encoder: ImageEmbeddingModel,
        fusion_head: FusionHead,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_head = fusion_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)
        return self.fusion_head(text_features, image_features)


def load_tokenizer_and_llama(config: Config, dtype: torch.dtype) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(config.text_model, low_cpu_mem_usage=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    quantization_config = None
    if config.use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    llama = AutoModelForCausalLM.from_pretrained(
        config.text_model,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    llama.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return tokenizer, llama


def infer_llama_hidden_size(llama: AutoModelForCausalLM) -> int:
    if hasattr(llama.config, "hidden_size"):
        return int(llama.config.hidden_size)
    if hasattr(llama.config, "n_embd"):
        return int(llama.config.n_embd)
    raise ValueError("Could not infer hidden size from LLaMA config")


def freeze_base_models(model: FuseMD) -> None:
    for param in model.text_encoder.embedding.text_base.parameters():
        param.requires_grad = False
    for param in model.text_encoder.embedding.clf_head.parameters():
        param.requires_grad = True

    for param in model.image_encoder.embedding.parameters():
        param.requires_grad = False
    for param in model.image_encoder.embedding.head.parameters():
        param.requires_grad = True

    for param in model.fusion_head.parameters():
        param.requires_grad = True


def trainable_parameters(model: FuseMD) -> List[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def make_loaders(config: Config, tokenizer: AutoTokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
    language_path = Path(config.data_root) / config.language
    train_dataset = MemeDataset("train", language_path, tokenizer, config)
    dev_dataset = MemeDataset("dev", language_path, tokenizer, config)
    test_dataset = MemeDataset("test", language_path, tokenizer, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )
    return train_loader, dev_loader, test_loader


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "image": batch["image"].to(device=device, dtype=dtype),
        "label": batch["label"].to(device=device, dtype=dtype),
        "input_ids": batch["input_ids"].to(device=device),
        "attention_mask": batch["attention_mask"].to(device=device),
        "image_id": batch["image_id"].to(device=device),
    }


def run_epoch(
    model: FuseMD,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    optimizer: Optional[torch.optim.Optimizer] = None,
    description: str = "Training",
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    steps = 0

    iterator = tqdm(loader, unit="batch", desc=description)
    for batch in iterator:
        batch = move_batch(batch, device, dtype)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(batch["input_ids"], batch["attention_mask"], batch["image"])
        loss = criterion(logits, batch["label"])

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        steps += 1
        iterator.set_postfix(loss=total_loss / max(steps, 1))

    return total_loss / max(steps, 1)


@torch.no_grad()
def collect_probabilities(
    model: FuseMD,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: List[float] = []
    labels: List[float] = []
    image_ids: List[int] = []

    for batch in loader:
        batch = move_batch(batch, device, dtype)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["image"])
        batch_probs = torch.sigmoid(logits)
        probs.extend(batch_probs.detach().cpu().numpy().tolist())
        labels.extend(batch["label"].detach().cpu().numpy().tolist())
        image_ids.extend(batch["image_id"].detach().cpu().numpy().tolist())

    return np.asarray(probs), np.asarray(labels), np.asarray(image_ids)


def threshold_grid(config: Config) -> List[float]:
    values = []
    current = config.threshold_start
    while current <= config.threshold_stop + 1e-9:
        values.append(round(current, 10))
        current += config.threshold_step
    return values


def optimize_threshold(probs: np.ndarray, labels: np.ndarray, config: Config) -> Tuple[float, float, Dict[float, float]]:
    scores: Dict[float, float] = {}
    for threshold in threshold_grid(config):
        preds = (probs >= threshold).astype(float)
        scores[threshold] = f1_score(labels, preds, average="macro", zero_division=0)

    best_threshold = max(scores, key=scores.get)
    return best_threshold, scores[best_threshold], scores


def get_trainable_checkpoint(model: FuseMD) -> Dict[str, object]:
    vit_state = model.image_encoder.state_dict()
    vit_head_state = {key: value.cpu() for key, value in vit_state.items() if "head" in key}

    return {
        "text_clf_head_state_dict": {
            key: value.cpu()
            for key, value in model.text_encoder.embedding.clf_head.state_dict().items()
        },
        "vit_head_state_dict": vit_head_state,
        "fusion_head_state_dict": {
            key: value.cpu()
            for key, value in model.fusion_head.state_dict().items()
        },
    }


def load_trainable_checkpoint(model: FuseMD, checkpoint: Dict[str, object]) -> None:
    model.text_encoder.embedding.clf_head.load_state_dict(checkpoint["text_clf_head_state_dict"])

    vit_current = model.image_encoder.state_dict()
    vit_current.update(checkpoint["vit_head_state_dict"])
    model.image_encoder.load_state_dict(vit_current, strict=False)

    model.fusion_head.load_state_dict(checkpoint["fusion_head_state_dict"])


def build_model(config: Config, llama: AutoModelForCausalLM, dtype: torch.dtype, fusion: str) -> FuseMD:
    hidden_size = infer_llama_hidden_size(llama)
    text_classifier = LlamaTextClassifier(llama=llama, hidden_size=hidden_size)
    text_encoder = TextEmbeddingModel(text_classifier, embedding_dim=128)
    image_encoder = ImageEmbeddingModel(config.image_model, embedding_dim=128)
    fusion_head = FusionHead(fusion=fusion, embedding_dim=128)
    model = FuseMD(text_encoder=text_encoder, image_encoder=image_encoder, fusion_head=fusion_head)

    if dtype == torch.float16:
        model = model.half()

    freeze_base_models(model)
    return model


def make_output_dirs(config: Config) -> Dict[str, Path]:
    root = Path(config.output_root)
    paths = {
        "predictions": root / "predictions" / config.language / "fusion",
        "metrics": root / "predictions" / config.language / "fusion",
        "models": root / "trained_model" / config.language / "fusion",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_run_artifacts(
    config: Config,
    paths: Dict[str, Path],
    model: FuseMD,
    tokenizer: AutoTokenizer,
    fusion: str,
    initial_lr: float,
    final_lr: float,
    best_epoch: int,
    best_threshold: float,
    best_dev_f1: float,
    threshold_scores: Dict[float, float],
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_ids: np.ndarray,
    test_loss: float,
    test_macro_f1: float,
    test_accuracy: float,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"custom_{config.language}_llamavit_fusion_{fusion}"
        f"_lr{initial_lr}_epoch{best_epoch}_bs{config.batch_size}_{timestamp}"
    )

    predictions = (test_probs >= best_threshold).astype(float)
    predictions_df = pd.DataFrame(
        {
            "image_id": test_ids.astype(int),
            "probability": test_probs,
            "prediction": predictions,
            "true_label": test_labels,
        }
    )
    predictions_df.to_csv(paths["predictions"] / f"{run_name}.csv", index=False)

    checkpoint = {
        **get_trainable_checkpoint(model),
        "fusion_method": fusion,
        "learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_dev_macro_f1": best_dev_f1,
        "threshold_scores": threshold_scores,
        "test_macro_f1": test_macro_f1,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "config": asdict(config),
        "tokenizer_vocab_size": len(tokenizer),
    }
    torch.save(checkpoint, paths["models"] / f"{run_name}.pth")

    report = classification_report(test_labels, predictions, digits=5, zero_division=0)
    matrix = confusion_matrix(test_labels, predictions)

    metrics = {
        "run_name": run_name,
        "date_time": datetime.now().isoformat(),
        "fusion_method": fusion,
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_dev_macro_f1": best_dev_f1,
        "threshold_scores": threshold_scores,
        "test_macro_f1": test_macro_f1,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "config": asdict(config),
    }

    with open(paths["metrics"] / f"{run_name}.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(paths["metrics"] / f"{run_name}.txt", "w", encoding="utf-8") as file:
        file.write(f"Run: {run_name}\n")
        file.write(f"Date and time: {metrics['date_time']}\n")
        file.write(f"Fusion: {fusion}\n")
        file.write(f"Initial LR: {initial_lr}\n")
        file.write(f"Final LR: {final_lr}\n")
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best threshold: {best_threshold}\n")
        file.write(f"Dev macro-F1: {best_dev_f1:.5f}\n")
        file.write(f"Test macro-F1: {test_macro_f1:.5f}\n")
        file.write(f"Test accuracy: {test_accuracy:.5f}\n")
        file.write(f"Test loss: {test_loss:.5f}\n\n")
        file.write(str(matrix))
        file.write("\n\n")
        file.write(report)

    if config.save_confusion_matrix_png:
        display = ConfusionMatrixDisplay(matrix)
        display.plot()
        plt.title(f"{config.language} {fusion} confusion matrix")
        plt.savefig(paths["metrics"] / f"{run_name}_confusion_matrix.png", bbox_inches="tight", dpi=200)
        plt.close()

    print(f"Saved checkpoint: {paths['models'] / f'{run_name}.pth'}")
    print(f"Saved predictions: {paths['predictions'] / f'{run_name}.csv'}")
    print(f"Saved metrics: {paths['metrics'] / f'{run_name}.json'}")


def train_one_run(
    config: Config,
    tokenizer: AutoTokenizer,
    llama: AutoModelForCausalLM,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    fusion: str,
    learning_rate: float,
    paths: Dict[str, Path],
) -> None:
    print(f"\nStarting run, fusion={fusion}, learning_rate={learning_rate}")
    model = build_model(config, llama, dtype, fusion).to(device)

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate, eps=config.optimizer_eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_dev_loss = float("inf")
    best_epoch = 0
    final_lr = learning_rate
    epochs_without_improvement = 0
    best_state: Optional[Dict[str, object]] = None

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            dtype,
            optimizer=optimizer,
            description=f"Train epoch {epoch}",
        )
        with torch.no_grad():
            dev_loss = run_epoch(
                model,
                dev_loader,
                criterion,
                device,
                dtype,
                optimizer=None,
                description=f"Dev epoch {epoch}",
            )

        scheduler.step(dev_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f}, "
            f"dev_loss={dev_loss:.5f}, lr={current_lr:.8f}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            final_lr = current_lr
            epochs_without_improvement = 0
            best_state = get_trainable_checkpoint(model)
            print(f"New best model at epoch {epoch}, dev_loss={dev_loss:.5f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is not None:
        load_trainable_checkpoint(model, best_state)
        model.to(device)
        print(f"Loaded best trainable layers from epoch {best_epoch}")

    dev_probs, dev_labels, _ = collect_probabilities(model, dev_loader, device, dtype)
    best_threshold, best_dev_f1, threshold_scores = optimize_threshold(dev_probs, dev_labels, config)
    for threshold, score in threshold_scores.items():
        print(f"Threshold {threshold:.1f}: dev_macro_f1={score:.5f}")
    print(f"Best threshold={best_threshold}, dev_macro_f1={best_dev_f1:.5f}")

    test_probs, test_labels, test_ids = collect_probabilities(model, test_loader, device, dtype)
    test_preds = (test_probs >= best_threshold).astype(float)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_accuracy = float((test_preds == test_labels).mean())

    with torch.no_grad():
        test_loss = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            dtype,
            optimizer=None,
            description="Test loss",
        )

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Language: {config.language}")
    print(f"Fusion method: {fusion}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Final learning rate: {final_lr}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best threshold: {best_threshold}")
    print(f"Development macro-F1: {best_dev_f1:.5f}")
    print(f"Test macro-F1: {test_macro_f1:.5f}")
    print(f"Test accuracy: {test_accuracy:.5f}")
    print(f"Test loss: {test_loss:.5f}")
    print("=" * 60)
    print(classification_report(test_labels, test_preds, digits=5, zero_division=0))

    save_run_artifacts(
        config=config,
        paths=paths,
        model=model,
        tokenizer=tokenizer,
        fusion=fusion,
        initial_lr=learning_rate,
        final_lr=final_lr,
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_dev_f1=best_dev_f1,
        threshold_scores=threshold_scores,
        test_probs=test_probs,
        test_labels=test_labels,
        test_ids=test_ids,
        test_loss=test_loss,
        test_macro_f1=test_macro_f1,
        test_accuracy=test_accuracy,
    )

    del model
    torch.cuda.empty_cache()


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    dtype = resolve_dtype(config.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"8-bit quantization enabled: {config.use_8bit}")

    tokenizer, llama = load_tokenizer_and_llama(config, dtype)
    train_loader, dev_loader, test_loader = make_loaders(config, tokenizer)
    paths = make_output_dirs(config)

    for fusion in config.fusion_methods:
        for learning_rate in config.learning_rates:
            train_one_run(
                config=config,
                tokenizer=tokenizer,
                llama=llama,
                train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                device=device,
                dtype=dtype,
                fusion=fusion,
                learning_rate=learning_rate,
                paths=paths,
            )


if __name__ == "__main__":
    main()
