# 🤖 Codsoft Internship Projects

> A collection of end-to-end machine learning projects built during my internship at CodSoft — covering regression, binary classification, multi-class classification, and imbalanced learning.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat&logo=pandas)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebooks-yellow?style=flat&logo=googlecolab)
![Status](https://img.shields.io/badge/Internship-CodSoft-red?style=flat)
![Projects](https://img.shields.io/badge/Projects-4-brightgreen?style=flat)

---

## 📁 Projects

---

### 🚢 1. Titanic Survival Prediction

> **Can we predict who survived the Titanic disaster?**

Binary classification model predicting passenger survival using demographic and ticket data.

| Detail | Info |
|---|---|
| Algorithm | Logistic Regression + Random Forest |
| Dataset | [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic) |
| Type | Binary Classification |
| Target | Survived (0 = No, 1 = Yes) |

📂 [View Project](./Titanic-Survival-Prediction/)

---

### 🎬 2. Movie Rating Prediction

> **Can we predict IMDb ratings of Indian movies?**

Regression model predicting movie ratings from genre, director, cast, and metadata.

| Detail | Info |
|---|---|
| Algorithm | Multiple Linear Regression |
| Dataset | IMDb Indian Movies Dataset |
| Type | Regression |
| R² Score | **0.8032** |

📂 [View Project](./Movie-Rating-Prediction/)

---

### 🌸 3. Iris Flower Classification

> **Can we identify iris species from petal and sepal measurements alone?**

Multi-class classification identifying 3 iris species with near-perfect accuracy.

| Detail | Info |
|---|---|
| Algorithm | Support Vector Machine (SVM) |
| Dataset | [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) |
| Type | Multi-class Classification |
| Accuracy | **96.67%** |

📂 [View Project](./Iris-Flower-Classification/)

---

### 📈 4. Sales Prediction

> **How much will we sell if we spend X on advertising?**

Regression model predicting product sales from TV, Radio and Newspaper ad budgets.

| Detail | Info |
|---|---|
| Algorithm | Multiple Linear Regression |
| Dataset | [Kaggle Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset) |
| Type | Regression |
| R² Score | **0.8972** |

**Key insight:** TV has the strongest sales correlation (0.78). Newspaper spend has near-zero impact.

📂 [View Project](./Sales-Prediction-ML/)

---

### 💳 5. Credit Card Fraud Detection

> **Can we catch fraudulent transactions before the customer even notices?**

Binary classification model detecting fraud in a severely imbalanced dataset (577:1 ratio) — achieving F1 = 0.84 with only 5 false alarms per 56,962 transactions.

| Detail | Info |
|---|---|
| Algorithm | Random Forest (tuned, 200 trees) |
| Dataset | [Kaggle ULB Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Type | Binary Classification (Imbalanced) |
| F1 Score | **0.84** |
| Frauds Caught | **75 / 98** |
| False Alarms | **5 only** |

**Key techniques:** SMOTE oversampling, threshold tuning, Precision-Recall optimization.

📂 [View Project](./Credit-Card-Fraud-Detection/)

---

## 📊 Complete Project Summary

| # | Project | Type | Algorithm | Key Score |
|---|---|---|---|---|
| 1 | Titanic Survival | Binary Classification | Logistic Regression / RF | — |
| 2 | Movie Rating | Regression | Multiple Linear Regression | R² = 0.80 |
| 3 | Iris Flower | Multi-class Classification | SVM | Accuracy = 96.67% |
| 4 | Sales Prediction | Regression | Multiple Linear Regression | R² = 0.90 |
| 5 | Credit Card Fraud | Imbalanced Classification | Random Forest + SMOTE | F1 = 0.84 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML models & evaluation |
| imbalanced-learn | SMOTE for class imbalance |
| Google Colab | Cloud notebook environment |

---

## 🚀 Getting Started

```bash
# Clone this repository
git clone https://github.com/Zishaniitm/Codsoft-Internship-Projects.git

# Navigate into any project
cd Codsoft-Internship-Projects/Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```

---

## 👤 Author

**Zishan**
BCA Student | IIT Madras BS Data Science Student | CodSoft ML Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/zishan-ahmad-155b24239/)
[![GitHub](https://img.shields.io/badge/GitHub-Zishaniitm-black?style=flat&logo=github)](https://github.com/Zishaniitm)

---

*If you found this useful, drop a ⭐ on the repo!*
