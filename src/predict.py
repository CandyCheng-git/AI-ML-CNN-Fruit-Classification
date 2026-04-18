import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

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

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "fruit_classifier_final_best_model.keras"
IMAGE_SIZE = (100, 100)


def load_and_prepare_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.array(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def predict_image(model_path: Path, image_path: Path) -> None:
    model = keras.models.load_model(model_path)
    image_array = load_and_prepare_image(image_path)
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("
Top-3 probabilities:")

    top_3_indices = np.argsort(predictions)[-3:][::-1]
    for idx in top_3_indices:
        print(f"- {CLASS_NAMES[idx]}: {predictions[idx]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fruit image prediction using the saved Keras model.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the .keras model file. Defaults to model/fruit_classifier_final_best_model.keras",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    predict_image(model_path=model_path, image_path=image_path)


if __name__ == "__main__":
    main()
