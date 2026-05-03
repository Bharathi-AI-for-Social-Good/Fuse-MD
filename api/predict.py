import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from local_config import load_local_api_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a local prediction request to the Fuse-MD FastAPI server."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to upload.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text or OCR transcription for the meme.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional threshold override for this request.",
    )
    parser.add_argument(
        "--url",
        help="Prediction endpoint URL. Defaults to the local API URL from api/local_config.py.",
    )
    return parser.parse_args()


def default_predict_url() -> str:
    config = load_local_api_config()
    return f"http://{config.host}:{config.port}/predict"


def encode_multipart_form(
    *,
    text: str,
    image_path: Path,
    image_bytes: bytes,
    threshold: Optional[float],
) -> tuple[bytes, str]:
    boundary = f"fusemd-{uuid.uuid4().hex}"
    lines: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        lines.extend(
            [
                f"--{boundary}".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"),
                b"",
                value.encode("utf-8"),
            ]
        )

    add_field("text", text)
    if threshold is not None:
        add_field("threshold", str(threshold))

    lines.extend(
        [
            f"--{boundary}".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"'
            ).encode("utf-8"),
            b"Content-Type: application/octet-stream",
            b"",
            image_bytes,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ]
    )

    body = b"\r\n".join(lines)
    return body, boundary


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    if not image_path.is_file():
        print(f"Image path is not a file: {image_path}")
        return 1

    image_bytes = image_path.read_bytes()
    url = args.url or default_predict_url()
    body, boundary = encode_multipart_form(
        text=args.text,
        image_path=image_path,
        image_bytes=image_bytes,
        threshold=args.threshold,
    )

    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )

    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        print(json.dumps(payload, indent=2))
        return 0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"API request failed with status {exc.code}.")
        print(detail)
        return 1
    except urllib.error.URLError as exc:
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else url
        print(f"Could not connect to Fuse-MD API at {base_url}.")
        print("Start it first with: python api/run_api.py")
        print(f"Connection error: {exc.reason}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
