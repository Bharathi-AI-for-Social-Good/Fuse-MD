import os
from pathlib import Path


LOCAL_MODEL_ROOT_ENV = "FUSEMD_LOCAL_MODEL_ROOT"
DEFAULT_LOCAL_MODEL_ROOT = "local_models"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def configured_local_model_root(raw_root: str | None = None) -> Path:
    configured = raw_root or os.getenv(LOCAL_MODEL_ROOT_ENV, DEFAULT_LOCAL_MODEL_ROOT)
    return resolve_repo_path(configured)


def local_model_folder_name(model_id: str) -> str:
    return model_id.replace("\\", "__").replace("/", "__")


def expected_local_model_path(model_id: str, raw_root: str | None = None) -> Path:
    return configured_local_model_root(raw_root) / local_model_folder_name(model_id)


def looks_like_explicit_local_path(text_model_name: str) -> bool:
    normalized = text_model_name.replace("/", "\\")
    return (
        normalized.startswith(".\\")
        or normalized.startswith("..\\")
        or normalized.startswith("local_models\\")
        or "\\" in text_model_name
    )


def runtime_text_model_path(text_model_name: str, raw_root: str | None = None) -> Path:
    direct_candidate = resolve_repo_path(text_model_name)
    if direct_candidate.exists():
        return direct_candidate

    absolute_candidate = Path(text_model_name)
    if absolute_candidate.is_absolute():
        raise FileNotFoundError(f"Local text model path not found: {absolute_candidate}")
    if looks_like_explicit_local_path(text_model_name):
        raise FileNotFoundError(f"Local text model path not found: {direct_candidate}")

    expected_path = expected_local_model_path(text_model_name, raw_root)
    if expected_path.exists():
        return expected_path

    raise FileNotFoundError(
        "Local text model is missing for "
        f"'{text_model_name}'. Expected folder: {expected_path}. "
        f"Run: python api/setup_local_model.py --model {text_model_name}"
    )
