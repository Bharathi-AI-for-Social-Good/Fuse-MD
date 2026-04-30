import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import MemeDataset
from models import (
    build_model,
    get_trainable_checkpoint,
    load_tokenizer_and_llama,
    load_trainable_checkpoint,
    resolve_dtype,
    trainable_parameters,
)
from train import collect_probabilities, optimize_threshold, run_epoch


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


def make_loaders(cfg, tokenizer) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_root = resolve_template_path(cfg.dataset.data_root)
    common_kwargs = {
        "data_root": data_root,
        "language": cfg.dataset.language,
        "tokenizer": tokenizer,
        "image_id_col": cfg.dataset.image_id_col,
        "text_col": cfg.dataset.text_col,
        "label_col": cfg.dataset.label_col,
        "max_length": cfg.training.max_length,
    }

    train_dataset = MemeDataset(
        split=cfg.dataset.train_split,
        oversample_positive_train=cfg.training.oversample_positive_train,
        **common_kwargs,
    )
    dev_dataset = MemeDataset(
        split=cfg.dataset.dev_split,
        oversample_positive_train=0,
        **common_kwargs,
    )
    test_dataset = MemeDataset(
        split=cfg.dataset.test_split,
        oversample_positive_train=0,
        **common_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
    )
    return train_loader, dev_loader, test_loader


def make_output_dirs(cfg) -> Dict[str, Path]:
    output_root = resolve_template_path(cfg.dataset.output_root)
    paths = {
        "predictions": output_root / "predictions" / cfg.dataset.language / "fusion",
        "metrics": output_root / "predictions" / cfg.dataset.language / "fusion",
        "models": output_root / "trained_model" / cfg.dataset.language / "fusion",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_run_artifacts(
    cfg,
    paths: Dict[str, Path],
    model,
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
        f"fusemd_{cfg.dataset.language}_{fusion}"
        f"_lr{initial_lr}_epoch{best_epoch}_bs{cfg.training.batch_size}_{timestamp}"
    )

    predictions = (test_probs >= best_threshold).astype(float)
    predictions_path = paths["predictions"] / f"{run_name}.csv"
    metrics_json_path = paths["metrics"] / f"{run_name}.json"
    metrics_txt_path = paths["metrics"] / f"{run_name}.txt"
    checkpoint_path = paths["models"] / f"{run_name}.pth"

    np.savetxt(
        predictions_path,
        np.column_stack(
            [
                test_ids.astype(int),
                test_probs,
                predictions.astype(int),
                test_labels.astype(int),
            ]
        ),
        delimiter=",",
        header="image_id,probability,prediction,true_label",
        comments="",
    )

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
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(checkpoint, checkpoint_path)
    stable_checkpoint_path = resolve_template_path(cfg.dataset.finetuned_model_path)
    stable_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, stable_checkpoint_path)

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
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(metrics_json_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(metrics_txt_path, "w", encoding="utf-8") as file:
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

    if cfg.training.save_confusion_matrix_png:
        display = ConfusionMatrixDisplay(matrix)
        display.plot()
        plt.title(f"{cfg.dataset.language} {fusion} confusion matrix")
        plt.savefig(paths["metrics"] / f"{run_name}_confusion_matrix.png", bbox_inches="tight", dpi=200)
        plt.close()

    logger.info("Saved checkpoint: %s", checkpoint_path)
    logger.info("Updated default inference checkpoint: %s", stable_checkpoint_path)
    logger.info("Saved predictions: %s", predictions_path)
    logger.info("Saved metrics: %s", metrics_json_path)


def train_one_run(
    cfg,
    tokenizer,
    llama,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    fusion: str,
    learning_rate: float,
    paths: Dict[str, Path],
) -> None:
    logger.info("Starting run with fusion=%s learning_rate=%s", fusion, learning_rate)
    model = build_model(llama, cfg.model.image_model, fusion, cfg.model.embedding_dim).to(device)

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate, eps=cfg.training.optimizer_eps)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg.training.scheduler_factor,
        patience=cfg.training.scheduler_patience,
        threshold=cfg.training.scheduler_threshold,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_dev_loss = float("inf")
    best_epoch = 0
    final_lr = learning_rate
    epochs_without_improvement = 0
    best_state: Optional[Dict[str, object]] = None

    for epoch in range(1, cfg.training.max_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            description=f"Train epoch {epoch}",
        )
        with torch.no_grad():
            dev_loss = run_epoch(
                model,
                dev_loader,
                criterion,
                device,
                optimizer=None,
                description=f"Dev epoch {epoch}",
            )

        scheduler.step(dev_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %s train_loss=%.5f dev_loss=%.5f lr=%.8f",
            epoch,
            train_loss,
            dev_loss,
            current_lr,
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            final_lr = current_lr
            epochs_without_improvement = 0
            best_state = get_trainable_checkpoint(model)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.training.early_stopping_patience:
            logger.info("Early stopping at epoch %s", epoch)
            break

    if best_state is not None:
        load_trainable_checkpoint(model, best_state)
        model.to(device)

    dev_probs, dev_labels, _ = collect_probabilities(model, dev_loader, device)
    best_threshold, best_dev_f1, threshold_scores = optimize_threshold(
        dev_probs,
        dev_labels,
        cfg.training.threshold_start,
        cfg.training.threshold_stop,
        cfg.training.threshold_step,
    )

    test_probs, test_labels, test_ids = collect_probabilities(model, test_loader, device)
    test_preds = (test_probs >= best_threshold).astype(float)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_accuracy = float((test_preds == test_labels).mean())

    with torch.no_grad():
        test_loss = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            description="Test loss",
        )

    save_run_artifacts(
        cfg=cfg,
        paths=paths,
        model=model,
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.model.dtype)
    use_8bit = bool(cfg.model.use_8bit and device.type == "cuda")
    if cfg.model.use_8bit and device.type != "cuda":
        logger.warning("8-bit quantization requested, but CUDA is unavailable. Continuing without 8-bit.")

    tokenizer, llama = load_tokenizer_and_llama(cfg.model.text_model, use_8bit, dtype, device)
    train_loader, dev_loader, test_loader = make_loaders(cfg, tokenizer)
    paths = make_output_dirs(cfg)

    for fusion in cfg.model.fusion_methods:
        for learning_rate in cfg.training.learning_rates:
            train_one_run(
                cfg=cfg,
                tokenizer=tokenizer,
                llama=llama,
                train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                device=device,
                fusion=fusion,
                learning_rate=float(learning_rate),
                paths=paths,
            )


if __name__ == "__main__":
    main()
