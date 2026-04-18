# Portfolio Summary

## One-line pitch
Built a CNN-based fruit classification system for 14 image classes, with strong validation accuracy and fast inference, packaged as a lightweight computer vision workflow.

## What this proves

This project shows that I can:

- frame a business problem into an ML task
- inspect dataset quality before training
- apply augmentation and regularisation intentionally
- compare model behaviour across validation splits
- tune hyperparameters instead of accepting the first result
- package a trained model for reuse

## Strong talking points for interviews

### 1. Why I did not blindly choose a deeper model
A common mistake is assuming a larger network automatically means better results. In this project, I deliberately stayed with a lighter CNN because the dataset size and validation behaviour suggested a real overfitting risk.

### 2. How I judged the final model
I did not use training accuracy alone. I looked at validation accuracy, validation loss, macro F1-score, confusion matrix behaviour, and inference speed.

### 3. Why this matters in real work
In real engineering, the best model is not always the biggest one. It is the one that balances quality, stability, speed, and deployment practicality.

## Suggested resume bullet
Designed and evaluated a CNN-based fruit image classification pipeline for 14 classes, achieving 96.96% validation accuracy and 0.9643 macro F1-score while maintaining lightweight inference performance for practical deployment discussion.
