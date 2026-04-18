# AI-ML-CNN-Fruit-Classification

A CNN-based image classification project that identifies 14 fruit categories using TensorFlow/Keras, with data augmentation, hyperparameter tuning, cross-validation, and deployment-focused evaluation.

## Why this project matters

Manual fruit sorting is slow, inconsistent, and costly in agriculture. This project explores how a compact convolutional neural network (CNN) can classify 14 fruit categories from images with strong validation performance and fast inference, making it relevant for smart agriculture and lightweight deployment scenarios.

## Business objective

Build an image classification system that can:
- classify 14 fruit categories from RGB images
- reduce manual sorting effort
- generalise to unseen fruit images
- stay lightweight enough for practical deployment

## Project highlights

- 14-class fruit image classification
- TensorFlow / Keras CNN pipeline
- data augmentation for robustness
- repeated stratified validation splits
- hyperparameter comparison
- final model selection using validation accuracy and validation loss
- deployment-focused timing analysis

## Final results

Based on the final selected model from the notebook workflow:

- **Validation Accuracy:** 96.96%
- **Validation Loss:** 0.3065
- **Macro F1-score:** 0.9643
- **Inference Time:** ~0.0413 seconds per image
- **Model Size:** 98,254 parameters

## Tech stack

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- Pillow

## Repository structure

```text
AI-ML-CNN-Fruit-Classification/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ assets/
│  ├─ README_assets.md
│  ├─ confusion_matrix.png
│  ├─ training_curve.png
│  └─ sample_predictions.png
├─ docs/
│  └─ notebook_cleanup_checklist.md
├─ model/
│  └─ fruit_classifier_final_best_model.keras
├─ notebooks/
│  └─ Fruit_Classification_Assessmennt-1844721.ipynb
└─ src/
   ├─ train.py
   ├─ predict.py
   └─ demo_inference.py
```

## Dataset

The original project uses a train/test folder structure with 14 fruit classes.

Expected local structure:

```text
data/
├─ train/
│  ├─ Apple Granny Smith/
│  ├─ Apricot/
│  ├─ Avocado/
│  ├─ Banana/
│  ├─ Blueberry/
│  ├─ Cactus fruit/
│  ├─ Cherry/
│  ├─ Corn/
│  ├─ Kiwi/
│  ├─ Mango/
│  ├─ Orange/
│  ├─ Pineapple/
│  ├─ Strawberry/
│  └─ Watermelon/
└─ test/
   └─ ...
```

Do **not** push the full dataset to GitHub unless you are certain redistribution is allowed and the repository size stays reasonable.

## How to run inference

Predict one image:

```bash
python src/predict.py --image path/to/image.jpg --model model/fruit_classifier_final_best_model.keras
```

Run a small demo on a folder of sample images:

```bash
python src/demo_inference.py --input_dir assets/demo_images --model model/fruit_classifier_final_best_model.keras
```

## How to retrain

```bash
python src/train.py --train_dir data/train --img_size 100 --batch_size 32 --epochs 20
```

This script is a cleaned training baseline derived from the notebook so the project is easier to understand and reuse.

## What makes this portfolio-ready

This repository is structured to show more than a class submission:
- problem framing
- reproducible environment files
- deployable saved model
- reusable Python scripts
- measurable evaluation outcomes
- practical deployment considerations

## Recommended next improvements

- add evaluation images to `assets/`
- export notebook charts as PNG files for faster recruiter scanning
- add transfer learning comparison using MobileNetV2 or ResNet
- add a lightweight web demo with Gradio or Streamlit

## Note

The notebook remains the full research-style workflow. The `src/` scripts are the cleaner engineering layer intended to make the project easier for employers and collaborators to review.
