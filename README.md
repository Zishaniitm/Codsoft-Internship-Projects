# 🤖 Codsoft Internship Projects

> A collection of machine learning projects built during my internship at CodSoft.  
> Each project covers a real-world problem solved end-to-end — from EDA to model deployment.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat&logo=pandas)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebooks-yellow?style=flat&logo=googlecolab)
![Status](https://img.shields.io/badge/Internship-CodSoft-red?style=flat)

---

## 📁 Projects

---

### 🚢 1. Titanic Survival Prediction

> **Can we predict who survived the Titanic disaster?**

Binary classification model that predicts passenger survival using passenger data like age, sex, ticket class, and embarkation point.

| Detail | Info |
|---|---|
| Algorithm | Logistic Regression + Random Forest |
| Dataset | [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic) |
| Type | Binary Classification |
| Target | Survived (0 = No, 1 = Yes) |

**Key Highlights:**
- Handled missing values in `Age` and `Embarked` columns
- Feature engineered `Title` from passenger names
- Compared Logistic Regression vs Random Forest performance

📂 [View Project](./Titanic-Survival-Prediction/)

---

### 🎬 2. Movie Rating Prediction

> **Can we predict IMDb ratings of Indian movies?**

Regression model that predicts IMDb movie ratings based on genre, director, actors, votes, and release year.

| Detail | Info |
|---|---|
| Algorithm | Multiple Linear Regression |
| Dataset | [Kaggle: IMDb Indian Movies Dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) |
| Type | Regression |
| Target | IMDb Rating (0–10) |
| R² Score | **0.8032** |

**Key Highlights:**
- Extensive text preprocessing on Genre, Director, Actor columns
- Label encoding for categorical features
- Achieved 80% accuracy in predicting movie ratings

📂 [View Project](./Movie-Rating-Prediction/)

---

### 🌸 3. Iris Flower Classification

> **Can we classify iris flowers by species from measurements alone?**

Multi-class classification model that identifies Iris flower species (Setosa, Versicolor, Virginica) from sepal and petal measurements.

| Detail | Info |
|---|---|
| Algorithm | Support Vector Machine (SVM) |
| Dataset | [Iris Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris) |
| Type | Multi-class Classification |
| Target | Species (3 classes) |
| Accuracy | **96.67%** |

**Key Highlights:**
- Perfect class separation for Setosa species
- Visualized decision boundaries with pair plots
- SVM outperformed KNN and Decision Tree on this dataset

📂 [View Project](./Iris-Flower-Classification/)

---

### 📈 4. Sales Prediction Using ML

> **How much will we sell if we spend X on advertising?**

Regression model that predicts product sales based on advertising budgets across TV, Radio, and Newspaper channels. Helps businesses make smarter, data-driven marketing decisions.

| Detail | Info |
|---|---|
| Algorithm | Multiple Linear Regression |
| Dataset | [Kaggle — Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset) |
| Type | Regression |
| Target | Sales (units in thousands) |
| R² Score | **0.8972** |

**Key Highlights:**
- EDA revealed TV has the strongest sales correlation (0.78)
- Newspaper spend has near-zero impact — key business insight
- 75%+ predictions accurate within ±2,000 units
- Built a 9-plot visualization dashboard

📂 [View Project](./Sales-Prediction-ML/)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| Pandas & NumPy | Data manipulation and numerical computing |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning models and evaluation |
| Google Colab | Cloud-based notebook environment |

---

## 📊 Project Summary

| # | Project | Type | Algorithm | Score |
|---|---|---|---|---|
| 1 | Titanic Survival | Classification | Logistic Regression / Random Forest | — |
| 2 | Movie Rating | Regression | Multiple Linear Regression | R² = 0.80 |
| 3 | Iris Flower | Classification | SVM | Accuracy = 96.67% |
| 4 | Sales Prediction | Regression | Multiple Linear Regression | R² = 0.90 |

---

## 🚀 Getting Started

```bash
# Clone this repository
git clone https://github.com/Zishaniitm/Codsoft-Internship-Projects.git

# Navigate into any project folder
cd Codsoft-Internship-Projects/Sales-Prediction-ML

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```

Or open directly in **Google Colab** — each project folder contains a `.ipynb` notebook ready to run.

---

## 👤 Author

**Zishan**  
BCA Student | IIT Madras BS Data Science Student  
CodSoft ML Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Zishaniitm-black?style=flat&logo=github)](https://github.com/Zishaniitm)

---

## 📄 License

This repository is open source and available under the [MIT License](./LICENSE).

---

*If you found this useful, drop a ⭐ on the repo!*
