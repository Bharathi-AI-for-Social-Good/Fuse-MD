import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from local_config import load_local_api_config, repo_root, resolve_repo_path


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.local_store import expected_local_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face text model once into Fuse-MD's local_models store."
    )
    parser.add_argument(
        "--model",
        help="Hugging Face model id to download, for example VishnuPJ/MalayaLLM_7B_Base.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path to inspect and infer the required text model.",
    )
    parser.add_argument(
        "--root",
        help="Override the local model root. Defaults to api/local_config.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download into the target folder even if it already exists.",
    )
    return parser.parse_args()


def infer_text_model_from_checkpoint(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = checkpoint.get("config", {})
    if isinstance(checkpoint_cfg.get("model"), dict):
        checkpoint_cfg = checkpoint_cfg["model"]

    text_model = checkpoint_cfg.get("text_model")
    if not text_model:
        raise KeyError(f"Checkpoint does not define a text model: {checkpoint_path}")
    return str(text_model)


def choose_model_id(args: argparse.Namespace) -> str:
    if args.model:
        return str(args.model)

    if args.checkpoint:
        checkpoint_path = resolve_repo_path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return infer_text_model_from_checkpoint(checkpoint_path)

    config = load_local_api_config()
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(
            "No model id was provided and the default checkpoint is missing. "
            "Pass --model or --checkpoint."
        )
    return infer_text_model_from_checkpoint(config.checkpoint_path)


def resolved_root_arg(args: argparse.Namespace) -> str:
    if args.root:
        return str(resolve_repo_path(args.root))
    return str(load_local_api_config().local_model_root)


def verify_model_files(target_path: Path) -> None:
    required_files = [target_path / "config.json", target_path / "tokenizer_config.json"]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Downloaded model is incomplete at {target_path}. Missing: {', '.join(missing_files)}"
        )

    has_weights = any(
        (target_path / file_name).exists()
        for file_name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    if not has_weights:
        raise FileNotFoundError(
            f"Downloaded model is incomplete at {target_path}. Missing model weights."
        )

    for index_name in ("pytorch_model.bin.index.json", "model.safetensors.index.json"):
        index_path = target_path / index_name
        if not index_path.exists():
            continue

        with open(index_path, "r", encoding="utf-8") as file:
            index_payload = json.load(file)

        weight_map = index_payload.get("weight_map", {})
        shard_names = sorted(set(str(name) for name in weight_map.values()))
        missing_shards = [name for name in shard_names if not (target_path / name).exists()]
        if missing_shards:
            raise FileNotFoundError(
                f"Downloaded model is incomplete at {target_path}. Missing shard files: "
                f"{', '.join(missing_shards)}"
            )


def safe_remove_tree(target_path: Path, root_path: Path) -> None:
    resolved_target = target_path.resolve()
    resolved_root = root_path.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"Refusing to remove path outside local model root: {resolved_target}")
    shutil.rmtree(resolved_target)


def main() -> int:
    args = parse_args()
    model_id = choose_model_id(args)
    root_arg = resolved_root_arg(args)
    root_path = Path(root_arg)
    target_path = expected_local_model_path(model_id, root_arg)
    root_path.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and args.force:
        safe_remove_tree(target_path, root_path)

    if target_path.exists():
        verify_model_files(target_path)
        print(f"Local model already exists: {target_path}")
        print("Use --force to re-download it.")
        return 0

    print(f"Downloading model: {model_id}")
    print(f"Target folder: {target_path}")
    snapshot_download(repo_id=model_id, local_dir=str(target_path))
    verify_model_files(target_path)
    print("Local model download complete.")
    print(f"Saved to: {target_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
