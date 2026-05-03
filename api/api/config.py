import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def auto_checkpoint_path() -> Path:
    explicit = os.getenv("FUSEMD_CHECKPOINT")
    if explicit:
        return resolve_repo_path(explicit)

    preferred = resolve_repo_path("trained_model/malayalam/fusion/fusemd_best.pth")
    if preferred.exists():
        return preferred

    candidates = sorted(
        repo_root().glob("trained_model/*/fusion/*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return preferred


def env_optional_float(name: str) -> Optional[float]:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return None
    return float(raw_value)


def env_optional_int(name: str) -> Optional[int]:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return None
    return int(raw_value)


def env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path
    host: str
    port: int
    default_threshold: Optional[float]
    max_length: Optional[int]
    use_8bit: bool
    device: str


settings = Settings(
    checkpoint_path=auto_checkpoint_path(),
    host=os.getenv("FUSEMD_HOST", "0.0.0.0"),
    port=int(os.getenv("FUSEMD_PORT", "8000")),
    default_threshold=env_optional_float("FUSEMD_THRESHOLD"),
    max_length=env_optional_int("FUSEMD_MAX_LENGTH"),
    use_8bit=env_bool("FUSEMD_USE_8BIT", False),
    device=os.getenv("FUSEMD_DEVICE", "auto").strip().lower(),
)
