from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "image": batch["image"].to(device=device, dtype=torch.float32),
        "label": batch["label"].to(device=device, dtype=torch.float32),
        "input_ids": batch["input_ids"].to(device=device, dtype=torch.long),
        "attention_mask": batch["attention_mask"].to(device=device, dtype=torch.long),
        "image_id": batch["image_id"].to(device=device),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    description: str = "Training",
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    steps = 0

    iterator = tqdm(loader, unit="batch", desc=description)
    for batch in iterator:
        batch = move_batch(batch, device)

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
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: List[float] = []
    labels: List[float] = []
    image_ids: List[int] = []

    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["image"])
        batch_probs = torch.sigmoid(logits)
        probs.extend(batch_probs.detach().cpu().numpy().tolist())
        labels.extend(batch["label"].detach().cpu().numpy().tolist())
        image_ids.extend(batch["image_id"].detach().cpu().numpy().tolist())

    return np.asarray(probs), np.asarray(labels), np.asarray(image_ids)


def threshold_grid(start: float, stop: float, step: float) -> List[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def optimize_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
) -> Tuple[float, float, Dict[float, float]]:
    scores: Dict[float, float] = {}
    for threshold in threshold_grid(threshold_start, threshold_stop, threshold_step):
        preds = (probs >= threshold).astype(float)
        scores[threshold] = f1_score(labels, preds, average="macro", zero_division=0)

    best_threshold = max(scores, key=scores.get)
    return best_threshold, scores[best_threshold], scores
