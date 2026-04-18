# Fruit Classification CNN

A compact computer vision project that classifies 14 fruit categories from RGB images using a Convolutional Neural Network (CNN).

This repository is packaged as a **portfolio project**, not a raw university submission. The goal is to show an employer that I can structure an end-to-end machine learning workflow: problem framing, data checking, preprocessing, augmentation, training, evaluation, reproducibility, and inference packaging.

## Why this project matters

Manual fruit sorting is slow, inconsistent, and expensive in real agricultural workflows. This project tests whether a lightweight CNN can classify fruit images accurately enough to support smarter sorting and quality-control pipelines.

## Project highlights

- Built an image classification pipeline for **14 fruit classes**.
- Used **CNN + data augmentation + early stopping** to improve generalisation.
- Evaluated model quality using **validation accuracy, loss, macro F1-score, and confusion matrix**.
- Chose a lighter model design instead of blindly using a deeper architecture, to reduce overfitting risk and keep inference fast.
- Exported the final trained model as a `.keras` file for reuse.

## Fruit classes

- Apple Granny Smith
- Apricot
- Avocado
- Banana
- Blueberry
- Cactus fruit
- Cherry
- Corn
- Kiwi
- Mango
- Orange
- Pineapple
- Strawberry
- Watermelon

## Reported results

Based on the final evaluation recorded in the project report:

- Best fold validation accuracy: **96.96%**
- Validation loss: **0.3065**
- Macro F1-score: **0.9643**
- Approximate inference speed: **0.0413 s/image**

These numbers suggest the model is reasonably strong for a compact academic dataset and lightweight enough to be discussed for practical deployment scenarios.

## Repository structure

```text
fruit-classification-cnn/
├─ README.md
├─ PORTFOLIO_SUMMARY.md
├─ PROJECT_STRUCTURE.md
├─ data_README.md
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ notebooks/
│  └─ Fruit_Classification_Assessmennt.ipynb
├─ model/
│  └─ fruit_classifier_final_best_model.keras
├─ src/
│  ├─ predict.py
│  └─ demo_inference.py
└─ assets/
   └─ .gitkeep
```

## How to run

### 1) Create environment

Using pip:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate TECH3300-Ass2
```

### 2) Run prediction

```bash
python src/predict.py --image path/to/test_image.jpg
```

## Notes on data

The full training/test dataset is **not included** in this repo bundle.

That is intentional.

For a portfolio repository, pushing a large raw dataset usually makes the repo heavier, messier, and less professional. If the dataset has redistribution limits, uploading it can also be the wrong move. Keep the repo focused on code, notebook, trained model, and results.

## What I would improve next

- Replace the baseline CNN with **transfer learning** experiments using MobileNet or ResNet.
- Add a proper **train.py** pipeline separated from notebook cells.
- Add **saved evaluation plots** under `assets/` or `reports/figures/`.
- Add a lightweight **Streamlit or Gradio demo** for browser-based inference.
- Add **unit-tested preprocessing utilities** for cleaner production structure.

## Recruiter-facing takeaway

This project shows more than model training. It demonstrates:

- practical ML workflow thinking
- data sanity checking
- model comparison and tuning
- trade-off awareness between accuracy, overfitting, and inference cost
- reproducibility and cleaner GitHub presentation

## Author

[Candy Cheng](https://github.com/CandyCheng-git/)
