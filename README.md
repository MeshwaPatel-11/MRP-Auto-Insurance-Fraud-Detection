# Machine Learning-Based Auto Insurance Fraud Detection

**Major Research Project (MRP)** · Master of Science in Data Science and Analytics
**Author:** Meshwa Patel (Student ID: 501390663)

A supervised machine learning pipeline that detects fraudulent auto insurance claims from
structured customer, policy, and incident data, and explains its decisions well enough to
support a human investigator.

---

## Research Question

> *Can a machine learning model accurately identify fraudulent auto insurance claims using
> structured claim and policyholder data, and can the model's decisions be explained well
> enough to support a human investigator?*

---

## Notebooks

Run the notebooks **in order** (01 -> 02 -> 03), since each step depends on the previous one.
Notebook 04 is independent and can be run after 01.

| Notebook | What it does |
|----------|--------------|
| `01_data_cleaning.ipynb` | Cleans the dataset, handles missing values, builds the target |
| `02_eda.ipynb` | Exploratory data analysis and figures |
| `03_modelling.ipynb` | Compares 3 models (with/without SMOTE), SHAP, fairness check, risk score |
| `04_external_testing.ipynb` | Validates the method on independent datasets |

All files (notebooks, data, and figures) sit in the same folder, so the notebooks read the
CSVs directly by filename.

---

## Datasets

| File | Role | Target |
|------|------|--------|
| `car_insurance_fraud_dataset.csv` | Main training/validation/test dataset (30,000 claims) | `fraud_reported` |
| `car_insurance_clean.csv` | Cleaned data produced by notebook 01 | `fraud` |
| `insurance_claims (2).csv` | Transfer test + in-domain benchmark | `fraud_reported` |
| `fraud_oracle.csv` | Independent benchmark | `FraudFound_P` |
| `insurance_fraud_data (1).csv` | Second independent benchmark | `fraud reported` |

The main dataset is a public Kaggle auto insurance fraud dataset, with a class imbalance of
roughly 7.7 : 1 (about 11.5% of claims are fraudulent).

---

## How to run

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn jupyter
jupyter notebook
```

Then open the notebooks and run them in order. Figures are saved as PNG files in the same
folder.

---

## Method

Numeric features are standardised and categorical features one-hot encoded inside a single
column transformer. The data is split 70 / 15 / 15 (train / validation / test) in a stratified
way. Each of three models -- Logistic Regression, Random Forest, and XGBoost -- is trained
twice, once on the original data and once with SMOTE applied inside the pipeline, so the effect
of resampling is measured rather than assumed. The best model is selected by validation F1,
explained with SHAP (global and per-claim), and turned into a Low / Medium / High rule-based
risk score. A fairness check and external validation complete the workflow.

---

## Key results (honest summary)

- The fraud signal in the main dataset is **weak and diffuse**; incident severity is the
  single most useful feature.
- The best model (**Logistic Regression + SMOTE**) reached a test **AUC-ROC of 0.726** with
  high recall (~0.69) and low precision (~0.22) -- a **triage / ranking tool**, not an
  automatic accept/reject system.
- The rule-based **risk score** works: the High tier had roughly double the fraud rate of the
  Low tier (22.4% vs 8.3%).
- A **fairness check** found the model over-flags younger claimants relative to their true
  fraud rate.
- **External validation:** the same pipeline reached **AUC ~0.80-0.85** on the `fraud_oracle`
  dataset, showing the method is sound and that the modest main-dataset numbers reflect a
  weak-signal dataset rather than a flawed approach.
