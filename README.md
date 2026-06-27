# 🔧 Predictive Maintenance — Multi-Class Failure Detection

> An ensemble ML pipeline (XGBoost · LightGBM · CatBoost · RandomForest) with Optuna tuning, SMOTE resampling, and SHAP explainability — built on the AI4I 2020 Predictive Maintenance Dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Failure Classes](#failure-classes)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [SHAP Explainability](#shap-explainability)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Further Enhancements](#further-enhancements)
- [Known Limitations](#known-limitations)

---

## Overview

This project frames industrial machine failure prediction as a **multi-class classification problem** across 6 categories (No Failure + 5 failure modes). Rather than a binary "will it fail?" model, each sample is assigned to its specific failure type — enabling targeted, actionable maintenance decisions.

**Key design choices:**
- 4-model soft-voting ensemble tuned per-model via Optuna
- SMOTE oversampling to handle severe class imbalance
- Per-class SHAP values averaged across ensemble members to maintain explainability fidelity
- Feature engineering to surface physically meaningful signals (`Power`, `temperature_difference`)

---

## Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) — also available on [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

**File:** `ai4i2020.csv`

| Property | Value |
|---|---|
| Total Samples | 10,000 |
| Features (raw) | 8 |
| Features (engineered) | 10 |
| Train / Test Split | 80% / 20% (stratified) |
| Class Balance | Highly imbalanced (~97% No Failure) |

**Raw Features Used:**

| Feature | Description |
|---|---|
| `Type` | Machine quality variant: L / M / H |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | In-process temperature |
| `Rotational speed [rpm]` | Spindle rotational speed |
| `Torque [Nm]` | Applied torque |
| `Tool wear [min]` | Cumulative tool wear time |

---

## Failure Classes

| Class ID | Name | Description | Test Samples |
|---|---|---|---|
| 0 | No Failure | Normal operation | 1930 |
| 1 | TWF | Tool Wear Failure | 9 |
| 2 | HDF | Heat Dissipation Failure | 23 |
| 3 | PWF | Power Failure | 18 |
| 4 | OSF | Overstrain Failure | 16 |
| 5 | RNF | Random Failure | 4 |

> ⚠️ **Note:** TWF and RNF have very few test samples (9 and 4 respectively), making reliable evaluation of those classes statistically difficult. This is a dataset constraint.

---

## Project Pipeline

```
Raw CSV
  │
  ├─► Feature Engineering
  │     ├── temperature_difference = Process temp − Air temp
  │     └── Power = Torque × RPM × 2π / 60
  │
  ├─► Multi-Class Target Assignment
  │     └── First active failure flag → class ID (0–5)
  │
  ├─► Train / Test Split (80/20, stratified)
  │
  ├─► StandardScaler (fit on train only)
  │
  ├─► SMOTE Oversampling (on scaled training data)
  │     └── Fallback: sample_weight='balanced' if SMOTE fails
  │
  ├─► Optuna Hyperparameter Tuning (30 trials × 4 models)
  │     ├── RandomForest (CPU)
  │     ├── XGBoost (GPU)
  │     ├── LightGBM (GPU)
  │     └── CatBoost (GPU)
  │
  ├─► Soft-Voting Ensemble
  │
  ├─► Evaluation
  │     ├── Classification Report
  │     ├── Confusion Matrix
  │     └── Accuracy / Macro F1 / ROC-AUC (OVR)
  │
  └─► SHAP Explainability
        ├── Averaged SHAP values (RF + XGB + LGB)
        ├── Bar summary (global feature importance)
        └── Dot beeswarm plots per class
```

---

## Model Architecture

### Ensemble: Soft-Voting Classifier

Four individually tuned base models vote by averaging predicted probabilities:

| Model | Tuned Params | Compute |
|---|---|---|
| `RandomForestClassifier` | `n_estimators=106, max_depth=12, min_samples_split=2` | CPU |
| `XGBClassifier` | `n_estimators=257, max_depth=7, lr=0.271` | GPU (CUDA) |
| `LGBMClassifier` | `n_estimators=231, max_depth=6, lr=0.166` | GPU |
| `CatBoostClassifier` | `iterations=256, depth=4, lr=0.300` | GPU |

**Tuning:** Optuna (30 trials each), optimizing **macro F1** on a held-out 20% validation split of the SMOTE-resampled training data.

---

## Results

```
======================================================================
CLASSIFICATION REPORT
======================================================================
              precision    recall  f1-score   support

  No Failure       0.99      0.98      0.99      1930
         TWF       0.11      0.22      0.14         9
         HDF       0.92      0.96      0.94        23
         PWF       1.00      1.00      1.00        18
         OSF       0.88      0.94      0.91        16
         RNF       0.00      0.00      0.00         4

    accuracy                           0.98      2000
   macro avg       0.65      0.68      0.66      2000
weighted avg       0.99      0.98      0.98      2000
======================================================================

  Accuracy        : 0.9785
  Macro F1-Score  : 0.6628
  ROC-AUC (OVR)   : 0.9446
```

**Interpretation:**
- HDF, PWF, and OSF are detected reliably (F1 ≥ 0.91)
- TWF suffers from extremely low support (9 test samples)
- RNF (4 test samples) is effectively undetectable at this dataset scale — a known limitation of the source data
- ROC-AUC of 0.9446 indicates strong probabilistic discrimination overall

---

## SHAP Explainability

SHAP values were computed using `TreeExplainer` on three of the four ensemble models (RF, XGBoost, LightGBM) and averaged to reflect soft-voting logic.

### Key per-class insights:

| Class | Top Drivers | Physical Interpretation |
|---|---|---|
| No Failure | `Tool wear min` (low), `Power` | Low wear = low failure risk |
| TWF | `Tool wear min` (high), `Type` | Worn tools of certain grade types fail |
| HDF | `temperature_difference`, `Rotational speed rpm` | Heat dissipation linked to temp delta and speed |
| PWF | `Power`, `Torque Nm` | Power failures driven directly by power load |
| OSF | `Tool wear min`, `Torque Nm` | Overstrain at high wear + high torque |
| RNF | No dominant signal | Random by nature — as expected |

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for XGBoost/LightGBM/CatBoost GPU modes)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/predictive-maintenance-ensemble.git
cd predictive-maintenance-ensemble

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
imbalanced-learn
shap
optuna
matplotlib
seaborn
jupyter
```

### Dataset Setup

Download `ai4i2020.csv` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) and place it in the project root directory.

```
predictive-maintenance-ensemble/
├── ai4i2020.csv           ← place here
├── predictive-maintenance-xgboost-refactored.ipynb
└── ...
```

If running on Kaggle, the dataset is auto-detected from the Kaggle input path.

---

## Usage

### Run in Jupyter

```bash
jupyter notebook predictive-maintenance-xgboost-refactored.ipynb
```

### Notebook Execution Order

The notebook is organized into sequential steps — run cells top to bottom:

| Step | What it does |
|---|---|
| Step 1 | Imports, data loading, feature engineering |
| Step 2 | Multi-class target creation, train/test split, scaling |
| Step 3 | SMOTE oversampling |
| Step 2A | Optuna hyperparameter tuning (run once, then paste results) |
| Step 2B | Build and train the 4-model soft-voting ensemble |
| Step 4 | Predictions, classification report, confusion matrix |
| Step 5 | SHAP explainability — summary bar and per-class dot plots |

> **Tip:** Step 2A (Optuna tuning) runs 120 trials total (30 × 4 models) and takes ~10–30 minutes depending on hardware. After it completes, paste the printed `best_params` dictionaries into Step 2B and you won't need to re-run tuning again.

---

## Project Structure

```
predictive-maintenance-ensemble/
│
├── ai4i2020.csv                                         # Dataset (download separately)
├── predictive-maintenance-xgboost-refactored.ipynb      # Main notebook
├── requirements.txt                                     # Python dependencies
├── README.md                                            # This file
│
└── outputs/                                             # Generated plots (optional)
    ├── summary.png                                      # SHAP bar summary (all classes)
    ├── output.png                                       # SHAP dot plot — No Failure
    ├── output1.png                                      # SHAP dot plot — TWF
    ├── output2.png                                      # SHAP dot plot — HDF
    ├── output3.png                                      # SHAP dot plot — PWF
    ├── output4.png                                      # SHAP dot plot — OSF
    └── output5.png                                      # SHAP dot plot — RNF
```

---

## Further Enhancements

See the [detailed enhancement section](#-further-enhancements-detail) below.

---

## Known Limitations

- **TWF and RNF have very few test samples** (9 and 4). F1 scores for these classes are unreliable at this scale — treat them as directional, not definitive.
- **CatBoost is excluded from SHAP averaging.** `TreeExplainer` on CatBoost inside a `VotingClassifier` requires extra handling; the current SHAP values reflect RF + XGB + LGB only. The explanations are still meaningful but don't perfectly mirror all four ensemble members.
- **GPU flags are hardcoded** (`device='cuda'`, `task_type='GPU'`). On CPU-only machines these will either fall back silently or raise errors depending on driver availability. Consider adding a runtime GPU detection check.
- **SMOTE on high-dimensional imbalanced data** can introduce synthetic noise for very small minority classes (4–9 samples). The generated samples may not reflect real failure distributions.

---

## 🔮 Further Enhancements (Detail)

### 1. 🎯 Per-Class Probability Threshold Tuning *(High Impact)*

Right now the model predicts the class with the highest probability (argmax). For rare failures like TWF, this almost always loses to "No Failure."

**Fix:** For each failure class, find the probability threshold that maximizes recall (or F1) on a validation set:

```python
from sklearn.metrics import precision_recall_curve

for cls_idx, cls_name in enumerate(failure_names[1:], start=1):
    probs = y_pred_proba[:, cls_idx]
    precision, recall, thresholds = precision_recall_curve(
        (y_test == cls_idx).astype(int), probs
    )
    # Pick threshold that balances precision and recall for your use case
```

This alone could dramatically improve TWF and RNF detection without changing the model at all.

---

### 2. 📊 Precision-Recall Curves Per Class *(Medium Impact)*

ROC-AUC is misleading on imbalanced data. Precision-Recall curves give a much more honest picture of how well the model handles rare failures.

```python
from sklearn.metrics import PrecisionRecallDisplay

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (cls_name, ax) in enumerate(zip(failure_names, axes.flatten())):
    PrecisionRecallDisplay.from_predictions(
        (y_test == i).astype(int),
        y_pred_proba[:, i],
        name=cls_name, ax=ax
    )
```

---

### 3. 🏗️ Add CatBoost to SHAP Averaging *(Low effort, Correctness fix)*

The current SHAP averaging excludes CatBoost. Fix it:

```python
cat_model = xgb_model.named_estimators_['cat']
shap_cat = shap.TreeExplainer(cat_model).shap_values(X_test_scaled)
std_cat = standardize_shap(shap_cat, num_classes)

avg_shap = (std_rf + std_xgb + std_lgb + std_cat) / 4.0
```

---

### 4. 🛡️ GPU Fallback Detection *(Portability fix)*

```python
import subprocess

def has_gpu():
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except:
        return False

DEVICE = 'cuda' if has_gpu() else 'cpu'
TASK_TYPE = 'GPU' if has_gpu() else 'CPU'
```

Then replace hardcoded `device='cuda'` and `task_type='GPU'` with these variables throughout.

---

### 5. 💰 Cost-Sensitive Learning *(High Business Impact)*

In predictive maintenance, **missing a failure is far more expensive than a false alarm** (unplanned downtime vs. an extra inspection). Encode this as a cost matrix:

```python
# Example: missing TWF costs 10x more than a false alarm
cost_matrix = {
    'TWF': 10,
    'HDF': 5,
    'PWF': 8,
    'OSF': 6,
    'RNF': 3,
    'No Failure': 1,
}

# Use these as class_weight in RF or scale_pos_weight in XGBoost
```

Then optimize threshold selection using expected cost rather than F1.

---

### 6. 🔄 Cross-Validation on Minority Classes *(Robustness)*

With only 9 TWF samples total, a single train/test split gives an unreliable picture. Use stratified k-fold:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    ensemble_model, X, y,
    cv=skf, scoring='f1_macro', n_jobs=-1
)
print(f"CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

### 7. 🕐 Time-Series Aware Splitting *(Data Integrity)*

The AI4I dataset has a `UDI` column (sequential unit ID) implying temporal order. Random splitting may allow future data to inform past predictions. Consider a time-based split:

```python
# Use first 80% of UDI as train, last 20% as test
split_idx = int(len(df) * 0.8)
df_sorted = df.sort_values('UDI')
train_df = df_sorted.iloc[:split_idx]
test_df  = df_sorted.iloc[split_idx:]
```

---

### 8. 🚀 Model Deployment *(Production Readiness)*

Package the trained ensemble as a REST API for real-time scoring:

```python
# Save the model + scaler together
import joblib

joblib.dump({'model': ensemble_model, 'scaler': scaler}, 'predictive_maintenance_model.pkl')

# Load and score new data
artifacts = joblib.load('predictive_maintenance_model.pkl')
new_data_scaled = artifacts['scaler'].transform(new_data)
predictions = artifacts['model'].predict(new_data_scaled)
failure_probs = artifacts['model'].predict_proba(new_data_scaled)
```

Wrap with FastAPI or Flask for a deployable endpoint.

---

### 9. 📈 Monitoring & Concept Drift Detection *(MLOps)*

Real industrial sensors drift over time — the model's calibration will degrade. Add a monitoring layer:

- **Distribution monitoring:** Track rolling mean/std of incoming feature values and alert when they shift beyond training distribution bounds.
- **Prediction drift:** Monitor the fraction of predictions per class over time — a sudden spike in any failure class warrants investigation.
- Libraries: [Evidently AI](https://www.evidentlyai.com/), [Alibi Detect](https://github.com/SeldonIO/alibi-detect)

---

### 10. 🔍 SHAP Interaction Values *(Deep Explainability)*

Beyond individual feature contributions, SHAP interaction values reveal which feature *pairs* drive predictions — useful for understanding failure root causes:

```python
# Computationally expensive — run on a sample
shap_interaction = shap.TreeExplainer(xgb_tree).shap_interaction_values(X_test_scaled[:100])
shap.summary_plot(shap_interaction, X_test_scaled[:100])
```

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

- Dataset: Stephan Matzka, *"Explainable Artificial Intelligence for Predictive Maintenance Applications"* — AI4I 2020 Predictive Maintenance Dataset
- SHAP: Lundberg & Lee, *"A Unified Approach to Interpreting Model Predictions"* (NeurIPS 2017)
- Optuna: Akiba et al., *"Optuna: A Next-generation Hyperparameter Optimization Framework"* (KDD 2019)