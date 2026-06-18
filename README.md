# Machine Learning-Based Auto Insurance Fraud Detection

**Major Research Project (MRP)** · Master of Science in Data Science and Analytics
**Author:** Meshwa Patel (Student ID: 501390663) · Toronto Metropolitan University

A supervised machine learning pipeline that detects fraudulent auto insurance claims from
structured customer, policy, and incident data, and explains its decisions well enough to
support a human investigator.

---

## Research Question

> *Can a machine learning model accurately identify fraudulent auto insurance claims using
> structured claim and policyholder data, and can the model's decisions be explained well
> enough to support a human investigator?*

---

## What this project does

1. **Cleans** a 30,000-claim auto insurance dataset (`01_data_cleaning`).
2. **Explores** where the fraud signal lives (`02_eda`).
3. **Models** fraud by comparing three classifiers, each with and without SMOTE, then
   explains the best model with SHAP and builds a transparent rule-based risk score
   (`03_modelling`).
4. **Validates** the method on three independent datasets (`04_external_testing`).

---

## Repository structure

```
.
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Clean the dataset, build the target
│   ├── 02_eda.ipynb                # Exploratory data analysis + figures
│   ├── 03_modelling.ipynb          # Compare models, SHAP, risk score, fairness
│   ├── 04_external_testing.ipynb   # Validate on 3 external datasets
│   └── *.py                        # Jupytext source (diff-friendly versions)
├── data/
│   ├── raw/                        # Source CSV files (main + test datasets)
│   └── car_insurance_clean.csv     # Output of notebook 01
├── figures/                        # All charts produced by the notebooks
├── docs/                           # Literature review, report, proposal alignment
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How to run

The notebooks are written in plain, well-commented Python and are meant to be run **in order**
(01 → 02 → 03), because each step depends on the previous one. Notebook 04 is independent.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/MRP-Auto-Insurance-Fraud-Detection.git
cd MRP-Auto-Insurance-Fraud-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and run the notebooks in order
jupyter notebook
```

All figures are written to the `figures/` folder as the notebooks run.

---

## Datasets

| File | Role | Rows | Target |
|------|------|------|--------|
| `car_insurance_fraud_dataset.csv` | **Main** training/validation/test | 30,000 | `fraud_reported` |
| `insurance_claims__2_.csv` | Transfer test + in-domain benchmark | 1,000 | `fraud_reported` |
| `fraud_oracle.csv` | Independent benchmark | 15,420 | `FraudFound_P` |
| `insurance_fraud_data__1_.csv` | Second independent benchmark | 12,002 | `fraud reported` |
| `insurance_data.csv` | Excluded (no fraud label) | 10,000 | — |
| `carclaims.csv` | Excluded (no fraud label) | 67,856 | — |

The main dataset is a public Kaggle auto insurance fraud dataset. Class imbalance is roughly
7.7 : 1 (about 11.5% of claims are fraudulent).

---

## Methodology

![Project methodology](figures/00_methodology_diagram.png)

Numeric features are standardised and categorical features one-hot encoded inside a single
column transformer. The data is split 70 / 15 / 15 (train / validation / test) in a stratified
way. Each of three models — Logistic Regression, Random Forest, XGBoost — is trained twice,
once on the original data and once with SMOTE applied inside the pipeline, so the effect of
resampling is measured rather than assumed. The best model is selected by validation F1,
explained with SHAP, and turned into a Low/Medium/High rule-based risk score.

---

## Key results (honest summary)

- The fraud signal in the main dataset is **weak and diffuse**; incident severity is the
  single most useful feature.
- The best model (**Logistic Regression + SMOTE**) reached a test **AUC-ROC of 0.726** with
  high recall (~0.69) and low precision (~0.22). It is a **triage / ranking tool**, not an
  automatic accept/reject system.
- The rule-based **risk score** works: the High tier had roughly double the fraud rate of the
  Low tier (22.4% vs 8.3%).
- A **fairness check** found the model over-flags younger claimants relative to their true
  fraud rate — reported openly.
- **External validation:** the same pipeline reached **AUC ~0.80–0.85** on the `fraud_oracle`
  dataset, showing the method is sound and that the modest main-dataset numbers reflect a
  weak-signal (likely synthetic) dataset rather than a flawed approach.

---

## Documentation

The `docs/` folder contains the literature review, the full MRP report, and a
proposal-alignment document mapping each proposal requirement to where it is delivered.

---

## License

Released under the MIT License — see [LICENSE](LICENSE).
