from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


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
        data_root: Path,
        language: str,
        tokenizer,
        image_id_col: str,
        text_col: str,
        label_col: str,
        max_length: int,
        oversample_positive_train: int = 0,
    ) -> None:
        self.split = split
        self.split_path = data_root / language / split
        self.tokenizer = tokenizer

        csv_path = self.split_path / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")

        df = pd.read_csv(csv_path).dropna(axis=0).copy()
        required_cols = {image_id_col, text_col, label_col}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing_cols)}")

        df[image_id_col] = df[image_id_col].astype(int)
        df[label_col] = df[label_col].map(normalize_label)
        id_to_row = {int(row[image_id_col]): row for _, row in df.iterrows()}

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
            raw_records.append((image_id, str(row[text_col]), float(row[label_col])))

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
            if split == "train" and int(label) == 1 and oversample_positive_train > 0:
                repeat += oversample_positive_train

            for _ in range(repeat):
                self.images.append(image_stack[idx])
                self.labels.append(label)
                self.image_ids.append(image_id)
                texts.append(text)

        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
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
