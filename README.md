<a name="readme-top"></a>

# Auto Insurance Fraud Detection

> The goal of this research is to build a binary classifier model that can separate
> fraudulent auto insurance claims from legitimate ones, and to **explain** its predictions
> well enough to support a human investigator.

**Author:** Meshwa Patel (Student ID: 501390663)
**Program:** Master of Science in Data Science and Analytics — Major Research Project (MRP)

---

## Table of Contents

1. [About the Project](#about-the-project)
2. [Research Objective](#research-objective)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Getting Started](#getting-started)
5. [Data Workflow](#data-workflow)
   - [Dataset](#dataset)
   - [Data Cleaning](#data-cleaning)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Methodology](#methodology)
   - [Modelling](#modelling)
   - [Model Explainability (SHAP)](#model-explainability-shap)
   - [Fraud Risk Score](#fraud-risk-score)
   - [Fairness Check](#fairness-check)
   - [External Validation](#external-validation)
6. [Results Summary](#results-summary)
7. [Repository Structure](#repository-structure)
8. [License](#license)

---

## About the Project

Insurance fraud is dynamic and rarely follows a fixed pattern, which makes it hard to detect.
Fraudulent auto claims, ranging from mild exaggeration to fully staged accidents, cause large
financial losses for insurers, and those losses are passed back to honest customers as higher
premiums. The challenge is that a fraudulent claim is usually designed to look exactly like a
genuine one.

This project applies multiple machine learning techniques to classify claims as fraudulent or
legitimate. Before modelling, exploratory data analysis is used to understand how the
categorical and numerical features relate to the fraud label. Because the data is heavily
imbalanced (only about 11.5% of claims are fraudulent), the **Synthetic Minority Over-sampling
Technique (SMOTE)** is applied, and every model is trained both **with and without** SMOTE so
the effect of resampling can be measured rather than assumed.

Three classifiers — **Logistic Regression, Random Forest, and XGBoost** — are compared under
identical preprocessing. The best model is explained using **SHAP** (global and per-claim),
and its signals are turned into a transparent rule-based **Fraud Risk Score** for investigator
use. The approach is evaluated using **precision, recall, F1-score**, and the **Area Under the
ROC Curve (AUC)**.


## Research Objective

> *Can a machine learning model accurately identify fraudulent auto insurance claims using
> structured claim and policyholder data, and can the model's decisions be explained well
> enough to support a human investigator?*

The project compares several machine learning models and reports its conclusions using
**Recall** and **AUC Score** as the primary metrics, since the dataset is imbalanced.


## Evaluation Metrics

Because the data is imbalanced, **accuracy is misleading** — a model that predicts "never
fraud" would score about 88% accuracy while catching zero fraud. The project therefore relies
on:

- **Precision** — of all claims flagged as fraud, how many were actually fraud.
- **Recall** — of all real frauds, how many the model caught.
- **F1-Score** — the balance of precision and recall.
- **AUC-ROC** — how well the model ranks claims by risk. An AUC near 1.0 indicates strong
  separability; an AUC near 0.5 is no better than chance.

A high recall with an acceptable AUC is the key parameter for selecting the best model, since
in fraud detection a missed fraud is usually costlier than a false alarm that an investigator
can dismiss.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/Auto-Insurance-Fraud-Detection.git
   cd Auto-Insurance-Fraud-Detection
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter and run the notebooks **in order** (01 → 02 → 03); notebook 04 is
   independent:

   ```bash
   jupyter notebook
   ```

All figures are saved as PNG files in the project folder as the notebooks run.


## Data Workflow

### Dataset

The main dataset is a publicly available Kaggle **Car Insurance Fraud** dataset containing
**30,000 claims and 24 features** plus the target. It is highly imbalanced: **3,440 (11.5%)**
claims are fraudulent and **26,560 (88.5%)** are legitimate, a ratio of roughly 7.7 : 1.

| Column | Description |
| --- | --- |
| `policy_state` | State where the policy was issued |
| `policy_deductible` | First-pay deductible amount |
| `policy_annual_premium` | Annual premium paid by the insured |
| `insured_age` | Age of the insured person |
| `insured_sex` | Gender of the insured |
| `insured_education_level` | Education level of the insured |
| `insured_occupation` | Occupation category |
| `insured_hobbies` | Hobby of the insured |
| `incident_type` | Type of incident |
| `collision_type` | Type of collision |
| `incident_severity` | Severity of the damage |
| `authorities_contacted` | Authority contacted at the incident |
| `incident_state` / `incident_city` | Location of the incident |
| `incident_hour_of_the_day` | Hour the incident occurred |
| `number_of_vehicles_involved` | Number of vehicles involved |
| `bodily_injuries` | Number of bodily injuries |
| `witnesses` | Number of witnesses |
| `police_report_available` | Whether a police report was filed |
| `total_claim_amount` | Total claim amount |
| `fraud_reported` | **Target:** Y = fraudulent, N = legitimate |


### Data Cleaning

Cleaning was deliberately light, since the data was already well structured. The steps were:

- Filling the only column with missing values, `authorities_contacted` (25.2% missing), with
  the category `"None"` — interpreting a blank as "no authority contacted".
- Parsing `incident_date` into a proper date type.
- Standardising text categories by trimming whitespace.
- Creating a numeric target `fraud` (1/0) from `fraud_reported` (Y/N).

The data was then split in a **stratified** way into **70% training, 15% validation, and 15%
test**, preserving the fraud rate in each subset.

```python
x_train, x_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
x_valid, x_test, y_valid, y_test = train_test_split(
    x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
```

### Exploratory Data Analysis

The central finding of the EDA is that the **fraud signal is weak and diffuse** — no single
feature cleanly separates fraud from legitimate claims.

**Target distribution.** The strong class imbalance is the most important property of the data.

![Target distribution](/Figures/01_target_distribution.png)

**Customer features.** Demographic features show only mild relationships with fraud; Sales and
Clerk occupations are slightly above average, and fraudulent claimants are marginally younger.

![Customer fraud rates](/Figures/03_customer_fraud_rates.png)

**Incident features.** Incident severity is the clearest single predictor — "Total Loss" claims
are fraudulent about 14.8% of the time versus roughly 9.6% for "Minor Damage".

![Incident fraud rates](/Figures/05_incident_fraud_rates.png)

**Claim amounts.** Fraudulent claims tend to be slightly higher, but the distributions overlap
heavily.

![Claim amounts](/Figures/07_claim_amounts.png)

**Correlation.** No numeric feature is strongly correlated with the target, confirming that
fraud must be inferred from many small signals rather than one dominant variable.

![Correlation heatmap](/Figures/08_correlation_heatmap.png)

**Temporal and geographic.** Fraud rates show no strong seasonal pattern and only modest
variation across states.

![Monthly trend](/Figures/09_monthly_trend.png)
![Geographic](/Figures/11_geographic.png)


### Methodology

The overall workflow runs from raw data through cleaning, EDA, feature engineering, the
train/validation/test split, model training with and without SMOTE, evaluation, explainability,
a fairness check, and external validation.

![Project methodology](/Figures/00_methodology_diagram.png)

Numeric features are standardised and categorical features one-hot encoded inside a single
`ColumnTransformer`, so identical preprocessing applies to all splits and prevents data
leakage. SMOTE is applied **inside the pipeline** so synthetic minority examples are generated
only from the training folds.


### Modelling

Each of the three models was trained twice — on the original imbalanced data and with SMOTE —
giving six configurations evaluated on the validation set.

| Model | Data | Precision | Recall | F1 | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | Original | 0.364 | 0.016 | 0.030 | 0.698 |
| Logistic Regression | SMOTE | 0.207 | 0.612 | **0.310** | 0.695 |
| Random Forest | Original | 0.333 | 0.004 | 0.008 | 0.740 |
| Random Forest | SMOTE | 0.434 | 0.045 | 0.081 | 0.736 |
| XGBoost | Original | 0.440 | 0.099 | 0.161 | 0.731 |
| XGBoost | SMOTE | 0.431 | 0.103 | 0.166 | 0.731 |

The clearest pattern is the **effect of SMOTE on recall**: without it, every model has
near-zero recall (it predicts "not fraud" for almost everything); with SMOTE, recall rises
sharply. Notably, AUC barely changes, which shows SMOTE shifts the decision threshold rather
than improving the model's underlying ability to rank risk.

The best model by validation F1 is **Logistic Regression with SMOTE**. The confusion matrices
for all six configurations are shown below.

![All confusion matrices](/Figures/13b_confusion_all_models.png)

On the held-out **test set**, the best model achieved **Precision 0.222, Recall 0.686, F1 0.335,
and AUC-ROC 0.726**, consistent with validation (no overfitting).

![ROC and PR curves](/Figures/14_roc_pr_curves.png)

### Model Explainability (SHAP)

SHAP is used to open up the model. The global importance and beeswarm plots confirm that
incident severity, claim size, witness count, and age are among the most influential features.

![SHAP importance](/Figures/15_shap_importance.png)
![SHAP beeswarm](/Figures/16_shap_beeswarm.png)

Per-claim **waterfall plots** explain a single flagged claim in plain language, showing which
red flags pushed it toward being classified as fraud — exactly the kind of explanation an
investigator needs.
![SHAP waterfall](/Figures/shap_waterfall_claim1.png)
![SHAP waterfall](/Figures/shap_waterfall_claim2.png)


### Fraud Risk Score

A transparent, rule-based score (0–100) assigns points for human-readable red flags (for
example, total-loss severity, no witnesses, a very high claim, a young claimant) and sorts
claims into **Low / Medium / High** tiers. Fraud rates rise monotonically across the tiers,
from **8.3% (Low)** to **22.4% (High)**, validating the score as a triage tool.

![Risk tiers](/Figures/17_risk_tiers.png)

### Fairness Check

The best model's behaviour was broken down by demographic group. The model **over-flags younger
claimants** relative to their actual fraud rate (about 47% flagged vs a 13.6% real rate for
under-30s), which a real deployment would need to correct. Reporting this openly is part of
responsible model development.

![Fairness by group](/Figures/23_fairness_by_group.png)


### External Validation

To check whether the pipeline generalises, the same method was applied to independent datasets.
The strongest result, an **AUC-ROC of about 0.80–0.85** on the `fraud_oracle` dataset, shows
the pipeline performs well where the data carries genuine fraud signal.

![Benchmark ROC](/Figures/19_benchmark_roc.png)

This is the key cross-dataset evidence: the method is sound, and the modest results on the main
dataset reflect a **weak-signal (likely synthetic) dataset** rather than a flawed approach.


## Results Summary

- The fraud signal in the main dataset is **weak and diffuse**; incident severity is the single
  most useful feature.
- The best model (**Logistic Regression + SMOTE**) reached a test **AUC-ROC of 0.726** with high
  recall (~0.69) and low precision (~0.22). It is a **triage / ranking tool**, not an automatic
  accept/reject system.
- The rule-based **risk score** works: the High tier had roughly double the fraud rate of the
  Low tier.
- A **fairness check** found the model over-flags younger claimants.
- **External validation** reached **AUC ~0.80–0.85** on a stronger dataset, confirming the
  method is sound.


## Repository Structure

```
.
├── 01_data_cleaning.ipynb        # Clean the dataset, build the target
├── 02_eda.ipynb                  # Exploratory data analysis + figures
├── 03_modelling.ipynb            # Compare models, SHAP, fairness, risk score
├── 04_external_testing.ipynb     # Validate on independent datasets
├── *.csv                         # Datasets
├── *.png                         # Figures produced by the notebooks
├── requirements.txt
├── LICENSE
└── README.md
```


## License

Distributed under the MIT License. See `LICENSE` for details.

