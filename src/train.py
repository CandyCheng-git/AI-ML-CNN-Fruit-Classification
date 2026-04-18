"""
Clean training baseline for the fruit classification project.

Purpose:
- make the project look like engineering work instead of notebook-only coursework
- provide a reproducible baseline training pipeline
- keep the implementation compact and readable

This script does not reproduce every experiment from the notebook.
It gives a clear training path that matches the same project direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(input_shape: tuple[int, int, int], num_classes: int, learning_rate: float) -> tf.keras.Model:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN fruit classifier.")
    parser.add_argument("--train_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--img_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--output_model", type=Path, default=Path("model/fruit_classifier_retrained.keras"))
    parser.add_argument("--output_history", type=Path, default=Path("model/training_history.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_history.parent.mkdir(parents=True, exist_ok=True)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=args.validation_split,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8, 1.2),
        horizontal_flip=False,
        fill_mode="nearest",
    )

    train_generator = datagen.flow_from_directory(
        directory=str(args.train_dir),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        directory=str(args.train_dir),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    model = build_model(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=train_generator.num_classes,
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(args.output_model)

    with args.output_history.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    print(f"Saved model to: {args.output_model}")
    print(f"Saved training history to: {args.output_history}")
    print("Class mapping:")
    print(train_generator.class_indices)


if __name__ == "__main__":
    main()
