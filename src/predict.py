"""
Predict a fruit class from one image using the saved Keras model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    "Apple Granny Smith",
    "Apricot",
    "Avocado",
    "Banana",
    "Blueberry",
    "Cactus fruit",
    "Cherry",
    "Corn",
    "Kiwi",
    "Mango",
    "Orange",
    "Pineapple",
    "Strawberry",
    "Watermelon",
]


def load_and_prepare_image(image_path: Path, img_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fruit class from one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an input image.")
    parser.add_argument("--model", type=Path, default=Path("model/fruit_classifier_final_best_model.keras"))
    parser.add_argument("--img_size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = tf.keras.models.load_model(args.model)
    image_array = load_and_prepare_image(args.image, args.img_size)

    probs = model.predict(image_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    print(f"Predicted class : {pred_label}")
    print(f"Confidence      : {confidence:.4f}")

    print("\nTop-3 predictions:")
    top3_idx = np.argsort(probs)[::-1][:3]
    for idx in top3_idx:
        print(f"- {CLASS_NAMES[int(idx)]}: {float(probs[idx]):.4f}")


if __name__ == "__main__":
    main()
