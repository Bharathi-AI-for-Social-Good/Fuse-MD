from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Update this once to point at the checkpoint you want the easy-mode API to use.
CHECKPOINT_PATH = (
    "trained_model/malayalam/fusion/"
    "custom_malayalam_llamavit_fusion_gated_lr1e-05_epoch7_bs16_20260426_002623.pth"
)
HOST = "127.0.0.1"
PORT = 8000
DEVICE = "auto"
THRESHOLD = None
MAX_LENGTH = None


@dataclass(frozen=True)
class LocalAPIConfig:
    checkpoint_path: Path
    host: str
    port: int
    device: str
    threshold: Optional[float]
    max_length: Optional[int]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def load_local_api_config() -> LocalAPIConfig:
    return LocalAPIConfig(
        checkpoint_path=resolve_repo_path(CHECKPOINT_PATH),
        host=str(HOST),
        port=int(PORT),
        device=str(DEVICE).strip().lower(),
        threshold=None if THRESHOLD is None else float(THRESHOLD),
        max_length=None if MAX_LENGTH is None else int(MAX_LENGTH),
    )
