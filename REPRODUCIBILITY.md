# Reproducibility guide

## 1. Environment

Create a fresh Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Run the notebook

Start Jupyter and execute `model.ipynb` from top to bottom:

```bash
jupyter lab model.ipynb
```

Keep `训练基因信息_features.xlsx` in the same directory as the notebook.

## 3. Main reproducibility settings

The analysis uses fixed random seeds where applicable, including `random_state=42` for data shuffling, stratified data splitting, Isolation Forest, and model fitting/search procedures.

The notebook includes:

- consensus anomaly detection using IQR, Z-score, and Isolation Forest;
- primary-sequence feature extraction;
- incorporation of predicted secondary-structure descriptors;
- comparison of XGBoost, random forest, gradient boosting, SVR-RBF, and Lasso regression;
- XGBoost hyperparameter optimization with randomized search and cross-validation;
- held-out evaluation;
- secondary-structure feature ablation;
- gain-based feature importance and SHAP analyses.

## 4. Trained model

`xgboost_best_model.pkl` is the serialized trained model artifact supplied with the study. Loading Python pickle/joblib files can execute code; load this file only in a trusted environment and from a trusted copy of this repository.

## 5. Manuscript alignment

Before archival release (for example, Zenodo or a journal data repository), the notebook, serialized model, manuscript-reported hyperparameters, and manuscript-reported performance values should be checked to ensure they correspond to the same final analysis run.
