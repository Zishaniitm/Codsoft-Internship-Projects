# 💳 Credit Card Fraud Detection Using Machine Learning

> Detect fraudulent credit card transactions in real-time using ensemble machine learning — achieving F1 = 0.84 with only 5 false alarms per 56,962 transactions.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-red?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat&logo=pandas)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?style=flat&logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Problem Statement

Credit card fraud costs the global economy billions of dollars annually. With thousands of transactions happening every second, manual detection is impossible. This project builds a machine learning model that **automatically flags fraudulent transactions in milliseconds** — allowing banks to block fraud before money leaves the customer's account.

The core challenge: only **0.17% of transactions are fraud** (492 out of 284,807) — a 577:1 class imbalance that makes standard ML approaches fail completely.

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── data/
│   └── README.md              ← Download instructions (dataset too large for GitHub)
│
├── model/
│   ├── logistic_regression.pkl
│   ├── random_forest_original.pkl
│   ├── random_forest_tuned.pkl    ← Best model
│   └── best_threshold.txt         ← Optimal decision threshold (0.79)
│
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb
│
├── outputs/plots/
│   ├── 01_class_distribution.png
│   ├── 02_amount_comparison.png
│   ├── 03_time_analysis.png
│   ├── 04_feature_discrimination.png
│   ├── 05_top_features_dist.png
│   ├── 06_correlation_with_fraud.png
│   ├── 07_smote_comparison.png
│   ├── 08_confusion_matrices.png
│   ├── 09_model_comparison.png
│   ├── 10_roc_curve.png
│   ├── 11_feature_importance.png
│   ├── 12_threshold_tuning.png
│   ├── 13_final_confusion_matrix.png
│   ├── 14_final_dashboard.png
│   ├── 15_precision_recall_curve.png
│   ├── 16_feature_importance_tuned.png
│   └── 17_caught_vs_missed.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Rows | 284,807 transactions |
| Columns | 31 (Time, V1–V28, Amount, Class) |
| Fraud | 492 (0.17%) |
| Genuine | 284,315 (99.83%) |
| Imbalance Ratio | 577:1 |
| Missing Values | None |

> ⚠️ Dataset not stored in this repo due to size (144MB). See `data/README.md` for download instructions.

### Column Description

| Column | Description |
|---|---|
| `V1–V28` | PCA-transformed features (anonymized for privacy) |
| `Time` | Seconds elapsed since first transaction |
| `Amount` | Transaction amount in Euros |
| `Class` | **0 = Genuine, 1 = Fraud** ← Target variable |

---

## 🔍 Key EDA Findings

1. **577:1 class imbalance** — standard accuracy is meaningless; must use Precision/Recall/F1
2. **Fraud clusters at lower amounts** — fraudsters probe cards with small transactions first
3. **Fraud has no sleep pattern** — spreads evenly across all hours; genuine drops at night
4. **V14, V17, V12 are top predictors** — most separated distributions between classes
5. **Zero missing values** — dataset is perfectly clean

---

## ⚙️ Methodology

```
1. Exploratory Data Analysis    →  17 visualizations across 6 analysis areas
2. Preprocessing               →  StandardScaler on Time & Amount
3. Train-Test Split            →  80/20 stratified (preserves 0.17% ratio)
4. Class Imbalance             →  SMOTE on training set only
                                  394 real frauds → 227,846 balanced samples
5. Model 1                     →  Logistic Regression (baseline)
6. Model 2                     →  Random Forest 100 trees (original)
7. Model 3                     →  Random Forest 200 trees, tuned (final)
8. Threshold Tuning            →  Optimal threshold search (0.01 → 0.99)
9. Evaluation                  →  Precision, Recall, F1, ROC-AUC
```

---

## 📏 Model Performance Comparison

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 | — |
| Random Forest (Original) | 0.43 | 0.88 | 0.58 | — |
| **Random Forest (Tuned) ✅** | **0.94** | **0.77** | **0.84** | **~0.97** |

### Final Model — Detailed Results (Threshold = 0.79)

| Metric | Score |
|---|---|
| Precision | **0.94** |
| Recall | **0.77** |
| F1 Score | **0.84** |
| Frauds Caught | **75 / 98** |
| False Alarms | **5 only** |
| Genuine Customers Disrupted | **0.009%** |

---

## 💡 Key Business Insights

1. **Auto-blocked:** 75 frauds caught instantly by the model
2. **Investigation pipeline:** Remaining 23 caught via customer complaints + manual review
3. **Near-zero disruption:** Only 5 false alarms per 56,962 transactions (99.99% of customers unaffected)
4. **V14 is the single most powerful fraud indicator** — confirmed by both EDA and feature importance
5. **SMOTE was critical** — F1 jumped from 0.11 → 0.84 after balancing

---

## 🚀 How to Run

### Setup
```bash
git clone https://github.com/Zishaniitm/Codsoft-Internship-Projects.git
cd Codsoft-Internship-Projects/Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

### Download Dataset
1. Go to [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place in `data/` folder

### Run Notebook
```bash
jupyter notebook notebooks/Credit_Card_Fraud_Detection.ipynb
```

### Load Best Model
```python
import pickle

with open("model/random_forest_tuned.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/best_threshold.txt", "r") as f:
    threshold = float(f.read())

# Predict on new transaction
prob = model.predict_proba(new_transaction)[0][1]
prediction = "🚨 FRAUD" if prob >= threshold else "✅ GENUINE"
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | 17 visualizations |
| Scikit-learn | Models, metrics, preprocessing |
| imbalanced-learn | SMOTE oversampling |
| Google Colab | Cloud notebook environment |

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
jupyter>=1.0.0
```

---

## 👤 Author

**Zishan**
BCA Student | IIT Madras BS Data Science Student | CodSoft ML Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](# 💳 Credit Card Fraud Detection Using Machine Learning

> Detect fraudulent credit card transactions in real-time using ensemble machine learning — achieving F1 = 0.84 with only 5 false alarms per 56,962 transactions.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-red?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat&logo=pandas)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?style=flat&logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Problem Statement

Credit card fraud costs the global economy billions of dollars annually. With thousands of transactions happening every second, manual detection is impossible. This project builds a machine learning model that **automatically flags fraudulent transactions in milliseconds** — allowing banks to block fraud before money leaves the customer's account.

The core challenge: only **0.17% of transactions are fraud** (492 out of 284,807) — a 577:1 class imbalance that makes standard ML approaches fail completely.

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── data/
│   └── README.md              ← Download instructions (dataset too large for GitHub)
│
├── model/
│   ├── logistic_regression.pkl
│   ├── random_forest_original.pkl
│   ├── random_forest_tuned.pkl    ← Best model
│   └── best_threshold.txt         ← Optimal decision threshold (0.79)
│
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb
│
├── outputs/plots/
│   ├── 01_class_distribution.png
│   ├── 02_amount_comparison.png
│   ├── 03_time_analysis.png
│   ├── 04_feature_discrimination.png
│   ├── 05_top_features_dist.png
│   ├── 06_correlation_with_fraud.png
│   ├── 07_smote_comparison.png
│   ├── 08_confusion_matrices.png
│   ├── 09_model_comparison.png
│   ├── 10_roc_curve.png
│   ├── 11_feature_importance.png
│   ├── 12_threshold_tuning.png
│   ├── 13_final_confusion_matrix.png
│   ├── 14_final_dashboard.png
│   ├── 15_precision_recall_curve.png
│   ├── 16_feature_importance_tuned.png
│   └── 17_caught_vs_missed.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Rows | 284,807 transactions |
| Columns | 31 (Time, V1–V28, Amount, Class) |
| Fraud | 492 (0.17%) |
| Genuine | 284,315 (99.83%) |
| Imbalance Ratio | 577:1 |
| Missing Values | None |

> ⚠️ Dataset not stored in this repo due to size (144MB). See `data/README.md` for download instructions.

### Column Description

| Column | Description |
|---|---|
| `V1–V28` | PCA-transformed features (anonymized for privacy) |
| `Time` | Seconds elapsed since first transaction |
| `Amount` | Transaction amount in Euros |
| `Class` | **0 = Genuine, 1 = Fraud** ← Target variable |

---

## 🔍 Key EDA Findings

1. **577:1 class imbalance** — standard accuracy is meaningless; must use Precision/Recall/F1
2. **Fraud clusters at lower amounts** — fraudsters probe cards with small transactions first
3. **Fraud has no sleep pattern** — spreads evenly across all hours; genuine drops at night
4. **V14, V17, V12 are top predictors** — most separated distributions between classes
5. **Zero missing values** — dataset is perfectly clean

---

## ⚙️ Methodology

```
1. Exploratory Data Analysis    →  17 visualizations across 6 analysis areas
2. Preprocessing               →  StandardScaler on Time & Amount
3. Train-Test Split            →  80/20 stratified (preserves 0.17% ratio)
4. Class Imbalance             →  SMOTE on training set only
                                  394 real frauds → 227,846 balanced samples
5. Model 1                     →  Logistic Regression (baseline)
6. Model 2                     →  Random Forest 100 trees (original)
7. Model 3                     →  Random Forest 200 trees, tuned (final)
8. Threshold Tuning            →  Optimal threshold search (0.01 → 0.99)
9. Evaluation                  →  Precision, Recall, F1, ROC-AUC
```

---

## 📏 Model Performance Comparison

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 | — |
| Random Forest (Original) | 0.43 | 0.88 | 0.58 | — |
| **Random Forest (Tuned) ✅** | **0.94** | **0.77** | **0.84** | **~0.97** |

### Final Model — Detailed Results (Threshold = 0.79)

| Metric | Score |
|---|---|
| Precision | **0.94** |
| Recall | **0.77** |
| F1 Score | **0.84** |
| Frauds Caught | **75 / 98** |
| False Alarms | **5 only** |
| Genuine Customers Disrupted | **0.009%** |

---

## 💡 Key Business Insights

1. **Auto-blocked:** 75 frauds caught instantly by the model
2. **Investigation pipeline:** Remaining 23 caught via customer complaints + manual review
3. **Near-zero disruption:** Only 5 false alarms per 56,962 transactions (99.99% of customers unaffected)
4. **V14 is the single most powerful fraud indicator** — confirmed by both EDA and feature importance
5. **SMOTE was critical** — F1 jumped from 0.11 → 0.84 after balancing

---

## 🚀 How to Run

### Setup
```bash
git clone https://github.com/Zishaniitm/Codsoft-Internship-Projects.git
cd Codsoft-Internship-Projects/Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

### Download Dataset
1. Go to [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place in `data/` folder

### Run Notebook
```bash
jupyter notebook notebooks/Credit_Card_Fraud_Detection.ipynb
```

### Load Best Model
```python
import pickle

with open("model/random_forest_tuned.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/best_threshold.txt", "r") as f:
    threshold = float(f.read())

# Predict on new transaction
prob = model.predict_proba(new_transaction)[0][1]
prediction = "🚨 FRAUD" if prob >= threshold else "✅ GENUINE"
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | 17 visualizations |
| Scikit-learn | Models, metrics, preprocessing |
| imbalanced-learn | SMOTE oversampling |
| Google Colab | Cloud notebook environment |

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
jupyter>=1.0.0
```

---

## 👤 Author

**Zishan**
BCA Student | IIT Madras BS Data Science Student | CodSoft ML Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Zishaniitm-black?style=flat&logo=github)](https://github.com/Zishaniitm)

---

*If you found this useful, drop a ⭐ on the repo!*)
[![GitHub](https://img.shields.io/badge/GitHub-Zishaniitm-black?style=flat&logo=github)](https://github.com/Zishaniitm)

---

*If you found this useful, drop a ⭐ on the repo!*
