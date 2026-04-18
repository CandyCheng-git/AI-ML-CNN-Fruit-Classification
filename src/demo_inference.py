"""Simple helper script.

Usage:
    python src/demo_inference.py path/to/image.jpg
"""

import sys
from pathlib import Path
from predict import predict_image, DEFAULT_MODEL_PATH


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/demo_inference.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    predict_image(DEFAULT_MODEL_PATH, image_path)


if __name__ == "__main__":
    main()
