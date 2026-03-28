"""Fine-tune YOLOv8n on custom GUI screenshots.

This script trains a YOLOv8n model on a labelled dataset of GUI screenshots,
producing weights that can be used with :class:`screenwalker.vision.detector.UIElementDetector`.

Quick start::

    # 1. Prepare dataset (Roboflow export or manual labelling)
    #    Expected layout:
    #      data/
    #        images/train/  *.png / *.jpg
    #        images/val/    *.png / *.jpg
    #        labels/train/  *.txt  (YOLO format)
    #        labels/val/    *.txt
    #        data.yaml      (classes, paths)

    # 2. Run training
    python scripts/train_detector.py --data data/data.yaml --epochs 50

    # 3. Use the trained model
    #    Best weights are saved to runs/detect/train/weights/best.pt

Requires the ``yolo`` extra::

    pip install screenwalker[yolo]

Dataset sources
---------------
* **Roboflow GUI dataset** — search for "GUI elements" or "UI components" on
  https://roboflow.com/universe
* **VINS dataset** — Rico/VINS mobile UI components, available on GitHub
* **Self-labelled** — capture your target application, annotate with
  `labelImg <https://github.com/HumanSignal/labelImg>`_ or Roboflow Annotate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "runs/detect"
DEFAULT_NAME = "screenwalker"


def train(
    data: str,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMG_SIZE,
    batch: int = DEFAULT_BATCH,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_NAME,
    device: str = "",
) -> Path:
    """Fine-tune a YOLO model on a GUI screenshot dataset.

    Args:
        data: Path to ``data.yaml`` describing the dataset.
        model: Base weights to start from (pretrained YOLOv8n by default).
        epochs: Number of training epochs.
        imgsz: Input image size (pixels, square).
        batch: Batch size (-1 for auto).
        project: Output directory for training runs.
        name: Sub-directory name inside *project*.
        device: Training device string (``"cpu"``, ``"0"``, ``"0,1"``).
            Empty string lets ultralytics auto-select.

    Returns:
        Path to the best weights file (``<project>/<name>/weights/best.pt``).

    Raises:
        ImportError: If ``ultralytics`` is not installed.
        FileNotFoundError: If *data* does not exist.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for training.\n"
            "Install with: pip install screenwalker[yolo]"
        ) from exc

    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print(f"Loading base model: {model}")
    yolo = YOLO(model)

    train_kwargs: dict = dict(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
    )
    if device:
        train_kwargs["device"] = device

    print(f"Starting training — {epochs} epochs, imgsz={imgsz}, batch={batch}")
    yolo.train(**train_kwargs)

    best = Path(project) / name / "weights" / "best.pt"
    if best.exists():
        print(f"\nTraining complete. Best weights: {best}")
    else:
        print(f"\nTraining complete. Check {Path(project) / name} for outputs.")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on GUI screenshots for screenwalker."
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Base weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help=f"Image size (default: {DEFAULT_IMG_SIZE})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"Output dir (default: {DEFAULT_PROJECT})")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"Run name (default: {DEFAULT_NAME})")
    parser.add_argument("--device", default="", help="Device: cpu, 0, 0,1 (default: auto)")
    args = parser.parse_args()

    try:
        train(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
        )
    except (ImportError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
