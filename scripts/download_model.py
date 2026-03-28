"""Download OmniParser V2 detection model from Hugging Face Hub.

Usage::

    python scripts/download_model.py
    python scripts/download_model.py --output-dir models/

The script downloads ``icon_detect/best.pt`` from
``microsoft/OmniParser-v2.0`` and places it at
``models/icon_detect/best.pt`` (the default path expected by
:func:`screenwalker.vision.detector.create_detector`).

Requires ``huggingface_hub``::

    pip install huggingface_hub
    # or
    pip install screenwalker[yolo]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ID = "microsoft/OmniParser-v2.0"
FILENAME = "icon_detect/best.pt"
DEFAULT_OUTPUT_DIR = "models"


def download(output_dir: str = DEFAULT_OUTPUT_DIR) -> Path:
    """Download the OmniParser V2 detection weights.

    Args:
        output_dir: Directory where ``icon_detect/best.pt`` will be saved.

    Returns:
        Path to the downloaded weights file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the model.\n"
            "Install with: pip install huggingface_hub"
        ) from exc

    dest_dir = Path(output_dir) / "icon_detect"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "best.pt"

    if dest_path.exists():
        print(f"Model already present at {dest_path} — skipping download.")
        return dest_path

    print(f"Downloading {REPO_ID}/{FILENAME} …")
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(Path(output_dir)),
    )
    print(f"Downloaded to {cached}")
    return Path(cached)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OmniParser V2 icon detection weights."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the model (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    try:
        path = download(args.output_dir)
        print(f"\nModel ready: {path}")
        print("You can now run scenarios with YOLO detection enabled.")
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
