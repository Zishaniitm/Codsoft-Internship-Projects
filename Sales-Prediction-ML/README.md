# 📈 Sales Prediction Using Machine Learning

> Predict product sales from advertising budgets across TV, Radio, and Newspaper channels using Multiple Linear Regression.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat&logo=pandas)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?style=flat&logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Problem Statement

Businesses spend thousands of dollars on advertising across multiple platforms — TV, Radio, and Newspapers — but have no reliable way to measure which channel drives the most sales. This project builds a machine learning model that **predicts future sales** based on advertising budgets, enabling data-driven marketing decisions.

---

## 📂 Project Structure

```
Sales-Prediction-ML/
│
├── data/
│   └── advertising.csv          ← Dataset (200 rows × 4 columns)
│
├── notebooks/
│   └── Sales_Prediction_ML.ipynb  ← Full Jupyter/Colab notebook
│
├── outputs/
│   └── plots/
│       ├── 01_distributions.png
│       ├── 02_scatter_plots.png
│       ├── 03_heatmap.png
│       ├── 04_boxplots.png
│       ├── 05_evaluation.png
│       ├── 06_residuals_vs_fitted.png
│       ├── 07_feature_coefficients.png
│       ├── 08_error_analysis.png
│       └── 09_summary_dashboard.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset) |
| Rows | 200 |
| Columns | 4 (TV, Radio, Newspaper, Sales) |
| Missing Values | None |
| Target Variable | `Sales` (units sold in thousands) |

### Column Description

| Column | Type | Description |
|---|---|---|
| `TV` | Feature (Input) | Advertising budget spent on TV ($000s) |
| `Radio` | Feature (Input) | Advertising budget spent on Radio ($000s) |
| `Newspaper` | Feature (Input) | Advertising budget spent on Newspaper ($000s) |
| `Sales` | Target (Output) | Units sold (in thousands) |

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **TV** has the strongest correlation with Sales → **0.78** 🔥
- **Radio** has moderate correlation with Sales → **0.58** ✅
- **Newspaper** has weak correlation with Sales → **0.23** ⚠️
- No missing values or duplicate rows found in the dataset
- Sales is roughly normally distributed — ideal for linear regression

---

## ⚙️ Methodology

```
1. Load & Explore Data       →  df.info(), df.describe(), df.isnull()
2. EDA & Visualization       →  Histograms, Scatter plots, Heatmap, Boxplots
3. Feature/Target Split      →  X = [TV, Radio, Newspaper]  |  y = Sales
4. Train-Test Split          →  80% training, 20% testing (random_state=42)
5. Model Training            →  LinearRegression().fit(X_train, y_train)
6. Prediction                →  model.predict(X_test)
7. Evaluation                →  MAE, MSE, RMSE, R² Score
8. Visualization             →  Residuals, Actual vs Predicted, Dashboard
```

### Model Equation (Learned)

```
Sales = 2.94 + 0.046 × TV + 0.189 × Radio + (−0.001) × Newspaper
```

**Interpretation:**
- Every **$1,000 extra on TV** → Sales increase by ~46 units
- Every **$1,000 extra on Radio** → Sales increase by ~189 units
- **Newspaper coefficient ≈ 0** → Negligible impact on sales

---

## 📏 Model Performance

| Metric | Score | Meaning |
|---|---|---|
| **R² Score (Test)** | **0.8972** | Model explains ~90% of variance in Sales |
| **R² Score (Train)** | **0.9022** | No overfitting — gap is only ~0.01 |
| **MAE** | **1.2178** | Average prediction error ≈ ±1,200 units |
| **MSE** | **2.9084** | Mean squared error |
| **RMSE** | **1.7054** | Typical error ≈ 1,700 units |

### Prediction Accuracy (Business-Friendly)

| Threshold | % of Predictions Within |
|---|---|
| ±1,000 units | ~52% |
| ±2,000 units | ~75% |
| ±3,000 units | ~90% |

---

## 💡 Key Business Insights

1. **TV advertising** is the single strongest driver of sales with 0.78 correlation
2. **Radio** provides the best ROI per dollar — high coefficient relative to typical spend
3. **Newspaper** budget has almost zero impact — reallocating it to TV/Radio would improve results
4. The model can predict sales for **any new advertising campaign** with ~90% accuracy

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open the notebook: `notebooks/Sales_Prediction_ML.ipynb`
2. Mount Google Drive and set your `BASE_PATH`
3. Upload `advertising.csv` to `data/` folder in Drive
4. Run all cells top to bottom

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/Zishaniitm/Codsoft-Internship-Projects.git
cd Codsoft-Internship-Projects/Sales-Prediction-ML

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/Sales_Prediction_ML.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Base plotting |
| Seaborn | Statistical visualizations |
| Scikit-learn | ML model, train-test split, metrics |
| Google Colab | Cloud notebook environment |

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
jupyter>=1.0.0
```

---

## 📁 Plots Generated

| Plot | Description |
|---|---|
| `01_distributions.png` | Histogram + KDE for all 4 variables |
| `02_scatter_plots.png` | Each feature vs Sales with regression line |
| `03_heatmap.png` | Correlation matrix heatmap |
| `04_boxplots.png` | Outlier detection via boxplots |
| `05_evaluation.png` | Actual vs Predicted + Residuals distribution |
| `06_residuals_vs_fitted.png` | Model health check |
| `07_feature_coefficients.png` | Feature impact on sales |
| `08_error_analysis.png` | Cumulative error distribution |
| `09_summary_dashboard.png` | Full project summary in one view |

---

## 👤 Author

**Zishan**
BCA Student | IIT Madras Data Science Student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/zishan-ahmad-155b24239/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Zishaniitm)

---

## 📄 License

This project is open source and available under the [MIT License](../LICENSE).

---

*If you found this useful, drop a ⭐ on the repo!*
