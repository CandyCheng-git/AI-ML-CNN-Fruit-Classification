# Project Structure

## Current repo

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
│  └─ Fruit_Classification_Assessmennt-1844721.ipynb
├─ model/
│  └─ fruit_classifier_final_best_model.keras
├─ src/
│  ├─ predict.py
│  └─ demo_inference.py
└─ assets/
   └─ .gitkeep
```

## Why this structure is better

### notebook/
Keeps the original analysis and training workflow visible.

### model/
Keeps the exported trained model in one predictable location.

### src/
Shows you understand code should move out of notebooks when preparing a project for reuse.

### assets/
Gives you a place to add sample predictions, confusion matrix screenshots, or demo images later.

## What not to dump into GitHub

Do not push these unless you are certain it is necessary and allowed:

- full raw dataset folders
- random cache files
- notebook checkpoints
- large temporary outputs
- duplicate model versions
- private tokens, API keys, or machine-specific paths
