"""
Run batch inference over a folder of images.
Useful for creating screenshots for the GitHub README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from predict import load_and_prepare_image, CLASS_NAMES
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch demo inference for fruit images.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("model/fruit_classifier_final_best_model.keras"))
    parser.add_argument("--img_size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = tf.keras.models.load_model(args.model)
    image_paths = [p for p in sorted(args.input_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    if not image_paths:
        print("No supported image files found.")
        return

    for image_path in image_paths:
        x = load_and_prepare_image(image_path, args.img_size)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        print(f"{image_path.name} -> {CLASS_NAMES[pred_idx]} ({float(probs[pred_idx]):.4f})")


if __name__ == "__main__":
    main()
