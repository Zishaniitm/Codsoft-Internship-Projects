# 🌸 Iris Flower Classification

> A machine learning project built during my internship at Codsoft.

---

## 📌 Problem Statement

The Iris flower dataset consists of three species — Setosa, Versicolor,
and Virginica. The objective is to train a machine learning model that
learns from sepal and petal measurements and accurately classifies
each flower into its correct species.

---

## 📊 Dataset

- **Source:** [Kaggle — Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- **Samples:** 150 flowers
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Missing Values:** None
- **Class Balance:** Perfectly balanced — 50 samples per class

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Data visualization |
| seaborn | Statistical plots |
| scikit-learn | ML models and evaluation |
| joblib | Model saving and loading |
| Google Colab | Development environment |

---

## 🔄 Project Pipeline

Raw CSV → EDA → Visualization → Preprocessing → Training → Evaluation → Saved Model

### Steps followed:
1. Loaded dataset from Kaggle CSV using pandas
2. Performed Exploratory Data Analysis (EDA)
3. Visualized data using countplot, histogram, boxplot, pairplot, heatmap
4. Split data — 80% training, 20% testing (stratified)
5. Applied StandardScaler for feature scaling
6. Trained 5 different ML models
7. Evaluated using accuracy, classification report, confusion matrix, cross validation
8. Saved best model using joblib

---

## 🤖 Models Trained & Results

| Model | Test Accuracy | CV Mean | CV Std |
|-------|--------------|---------|--------|
| Logistic Regression | 93.33% | 95.83% | ±2.64% |
| K-Nearest Neighbors | 93.33% | 95.83% | ±2.64% |
| Decision Tree | 93.33% | 95.00% | ±1.67% |
| Random Forest | 90.00% | 95.00% | ±3.12% |
| **SVM (RBF Kernel)** | **96.67%** | **96.67%** | **±1.67%** |

---

## 🏆 Best Model — Support Vector Machine (SVM)

- **Kernel:** RBF (Radial Basis Function)
- **Test Accuracy:** 96.67%
- **Why SVM won:** The RBF kernel creates curved decision boundaries
  that handle the slight overlap between Versicolor and Virginica
  better than any other model tested.

### Per-class Performance (SVM):

| Species | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Setosa | 100% | 100% | 100% |
| Versicolor | 100% | 90% | 94.7% |
| Virginica | 90.9% | 100% | 95.2% |

---

## 📁 Files in this Repository

| File | Description |
|------|-------------|
| `iris_classification.ipynb` | Complete Colab notebook |
| `IRIS.csv` | Raw dataset from Kaggle |
| `models/best_model_svm.pkl` | Trained SVM model |
| `models/scaler.pkl` | Fitted StandardScaler |

---

## 🚀 How to Run

1. Open `iris_classification.ipynb` in Google Colab
2. Upload `IRIS.csv` when prompted
3. Run all cells in order
4. Trained model auto-saves to `models/` folder

---

## 💡 Key Learnings

- Petal features (length + width) are far more discriminative than sepal features
- Setosa is linearly separable — any model classifies it perfectly
- Versicolor and Virginica overlap slightly — SVM handles this best
- Feature scaling is critical for SVM and KNN performance
- Cross validation gives more reliable performance estimates than single test split
