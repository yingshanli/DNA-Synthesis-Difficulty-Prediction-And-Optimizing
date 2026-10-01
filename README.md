# An interpretable machine-learning framework for predicting and optimizing DNA synthesis difficulty

This repository contains the data, analysis notebook, and trained model associated with the manuscript **“An interpretable machine learning framework for predicting and optimizing DNA synthesis difficulty.”**

## Overview

The study models DNA synthesis duration from sequence-derived features using real industrial synthesis records generated on a standardized production workflow. The analysis combines primary-sequence descriptors with predicted secondary-structure features, compares multiple regression algorithms, evaluates the incremental contribution of secondary-structure information, interprets the final model using SHAP, and explores synonymous sequence redesign as an in silico proof of concept.

## Repository contents

- `model.ipynb` — complete analysis notebook, including data preprocessing, feature extraction, model comparison, hyperparameter optimization, feature-ablation analysis, held-out evaluation, and SHAP interpretation.
- `训练基因信息_features.xlsx` — anonymized industrial DNA synthesis dataset used by the notebook. The workbook contains DNA sequence, sequence length, synthesis duration, and precomputed secondary-structure descriptors.
- `xgboost_best_model.pkl` — serialized trained model artifact supplied with the analysis.
- `requirements.txt` — Python package requirements.
- `DATA_DICTIONARY.md` — description of the input data columns.
- `REPRODUCIBILITY.md` — instructions for reproducing the analysis.
- `CITATION.cff` — citation metadata for this repository.
- `LICENSE` — MIT license for the repository code. Data use may be subject to separate institutional or company requirements.

## Data summary

The supplied workbook contains **1,677 production records before quality-control and anomaly filtering**. The analysis notebook applies quality control and removes records identified as anomalous by at least two of three approaches (IQR, Z-score, and Isolation Forest), yielding the curated dataset used for model development.

The main data columns are:

- `ID`
- `Sequence`
- `Length`
- `Duration`
- `Vienna Structure`
- `MFE (kcal/mol)`
- `Max Stem Length`
- `Avg Stem Length`
- `Total Loops`
- `Max Loop Size`
- `GU Pairs`
- `Pairing Rate`
- `Stem Count`

See `DATA_DICTIONARY.md` for details.

## Analysis workflow

1. Load and quality-control the industrial synthesis records.
2. Extract primary-sequence features, including sequence length, nucleotide composition, di-nucleotide composition, 3-mer composition, and repeat-related features.
3. Incorporate eight predicted secondary-structure descriptors.
4. Compare multiple regression algorithms.
5. Optimize the selected XGBoost model using cross-validation.
6. Evaluate the optimized model on a held-out subset.
7. Perform feature-ablation analysis to quantify the incremental predictive value of secondary-structure descriptors.
8. Interpret the fitted model using XGBoost gain-based importance and SHAP.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter lab model.ipynb
```

The notebook expects `训练基因信息_features.xlsx` to be in the repository root.

## Reproducibility

Fixed random seeds are used where applicable, including `random_state=42` for shuffling, stratified splitting, and model development. See `REPRODUCIBILITY.md` for details.

## Data and code availability

The dataset contains anonymized industrial synthesis records. No customer-identifying information is included in the supplied analysis file. Before public release, users should confirm that data sharing complies with all applicable company, institutional, contractual, and intellectual-property requirements.

## Citation

If you use this repository, please cite the associated manuscript when available. Citation metadata are provided in `CITATION.cff`.
