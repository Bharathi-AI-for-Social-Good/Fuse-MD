import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from local_config import load_local_api_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Fuse-MD FastAPI server using local easy-mode defaults."
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path to use instead of api/local_config.py.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Device override for the API server.",
    )
    parser.add_argument(
        "--host",
        help="Host override for the API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port override for the API server.",
    )
    return parser.parse_args()


def build_effective_config(
    args: argparse.Namespace,
) -> tuple[Path, str, int, str, Optional[str], Optional[str]]:
    config = load_local_api_config()
    checkpoint_path = (
        resolve_repo_path(args.checkpoint) if args.checkpoint else config.checkpoint_path
    )
    host = args.host or config.host
    port = args.port or config.port
    device = (args.device or config.device).strip().lower()
    threshold = None if config.threshold is None else str(config.threshold)
    max_length = None if config.max_length is None else str(config.max_length)
    return checkpoint_path, host, port, device, threshold, max_length


def configure_environment(
    checkpoint_path: Path,
    host: str,
    port: int,
    device: str,
    threshold: Optional[str],
    max_length: Optional[str],
) -> None:
    os.environ["FUSEMD_CHECKPOINT"] = str(checkpoint_path)
    os.environ["FUSEMD_HOST"] = host
    os.environ["FUSEMD_PORT"] = str(port)
    os.environ["FUSEMD_DEVICE"] = device

    if threshold is None:
        os.environ.pop("FUSEMD_THRESHOLD", None)
    else:
        os.environ["FUSEMD_THRESHOLD"] = threshold

    if max_length is None:
        os.environ.pop("FUSEMD_MAX_LENGTH", None)
    else:
        os.environ["FUSEMD_MAX_LENGTH"] = max_length


def main() -> int:
    args = parse_args()
    checkpoint_path, host, port, device, threshold, max_length = build_effective_config(args)

    if not checkpoint_path.exists():
        print("Fuse-MD API could not start.")
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Update CHECKPOINT_PATH in api/local_config.py or pass --checkpoint.")
        return 1

    configure_environment(checkpoint_path, host, port, device, threshold, max_length)
    base_url = f"http://{host}:{port}"
    print("Starting Fuse-MD API")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"URL: {base_url}")
    print(f"Docs: {base_url}/docs")

    from api.app import app

    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
