# Titanic Survival Prediction 🚢

> Built my first Machine Learning project and honestly it felt like being a detective 🕵️

A complete end-to-end binary classification project using real Titanic passenger data to predict survival outcomes — covering every stage of the ML pipeline from raw data to evaluated model.

---

## The Problem

The Titanic sank in 1912. Of the 891 passengers in this dataset, only 342 survived — a survival rate of just 38%.

Using real passenger data — age, gender, ticket class, fare paid, family size, and more — I built a machine learning model that predicts whether a passenger would have survived or not. This is a **binary classification problem**: the output is either `1` (survived) or `0` (did not survive).

---

## Dataset

**Source:** [Kaggle — Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

| Detail | Value |
|---|---|
| Total passengers | 891 |
| Features (original) | 12 |
| Target column | `Survived` (0 or 1) |
| Survival rate | 38% |
| Missing value columns | `Age` (177), `Cabin` (687), `Embarked` (2) |

---

## Project Pipeline

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation
```

### Phase 1 — Understanding the data
Loaded the dataset and inspected all 12 columns using `df.info()`, `df.describe()`, and `df.isnull().sum()`. Identified three columns with missing values and formed a plan for each before touching any model.

### Phase 2 — Exploratory Data Analysis (EDA)
Visualised survival patterns across every key feature. Key findings:
- **Gender:** 74% of women survived vs only 19% of men
- **Class:** 63% of 1st class survived vs 24% of 3rd class
- **Age:** Children under 12 had the highest survival rate (~58%)
- **Family size:** Small families (2–4) survived at 58%; solo travellers at only 30%

### Phase 3 — Data Preprocessing & Cleaning

The dataset had missing information for almost 200 passengers' ages and 687 cabin numbers — you can't just ignore blanks in ML, the model crashes.

| Problem | Solution |
|---|---|
| 177 missing `Age` values | Filled with **median age grouped by Pclass + Sex** — smarter than a single overall average |
| 687 missing `Cabin` values | Extracted `HasCabin` flag (1/0) before dropping the column |
| 2 missing `Embarked` values | Filled with mode (most common port: Southampton) |
| Text columns (`Sex`, `Embarked`) | Converted to numbers using binary encoding and one-hot encoding |
| Scale difference (`Age` vs `Fare`) | Applied `StandardScaler` — puts all features on the same ruler |

Some columns were text like `"male/female"` or `"S/C/Q"` — computers don't understand words, only numbers. Encoding converts these into a format models can actually read.

### Phase 4 — Feature Engineering

Raw data alone wasn't enough — features like `SibSp` and `Parch` sitting separately didn't mean much on their own. Created 6 brand new features from scratch:

| New Feature | Built From | What it captures |
|---|---|---|
| `Title` | `Name` column (regex) | Gender + age group + social class in one feature (Mr., Mrs., Master, Miss, Rare) |
| `FamilySize` | `SibSp + Parch + 1` | Total people travelling together |
| `IsAlone` | `FamilySize` | Binary flag — did they travel solo? |
| `AgeGroup` | `Age` (binned) | Child / Teen / YoungAdult / MiddleAge / Senior |
| `HasCabin` | `Cabin` null check | Whether a cabin number was recorded |
| `FareBand` | `Fare` (quantile bins) | Fare quartile (0–3) — removes outlier distortion |

These engineered features appeared in the **top 7 most important features** the Random Forest identified — validation that the extra effort paid off.

### Phase 5 — Model Training

Trained 4 different algorithms and compared them all:

| Model | Accuracy | Notes |
|---|---|---|
| **Random Forest** | **84%** | Best performer — ensemble of 100 decision trees |
| Logistic Regression | 80% | Strong linear baseline |
| Decision Tree | 79% | Interpretable but prone to overfitting |
| KNN | 76% | Distance-based, needs good scaling |

Also ran **5-fold cross-validation** on each model to get stable, reliable accuracy estimates rather than relying on a single train/test split.

### Phase 6 — Model Evaluation

Went beyond accuracy to get the full picture:

```
Confusion Matrix (Random Forest on test set):
              Predicted: 0    Predicted: 1
Actual: 0         96              13       ← 13 false alarms
Actual: 1         15              55       ← 15 survivors missed
```

| Metric | Score |
|---|---|
| Accuracy | 84.4% |
| Precision | 80.9% |
| Recall | 78.6% |
| F1 Score | 79.7% |
| AUC Score | 0.87 |

Also applied `GridSearchCV` to tune hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) — optimised for F1, not just accuracy.

---

## Results

Random Forest came out on top with **84% accuracy** — the model correctly predicted survival for 84 out of every 100 passengers. It figured out on its own that gender was the single biggest survival factor, followed by fare paid and age. Exactly what history tells us.

**Feature importance (top 5):**
```
Sex          ████████████████████  26%
Title        ████████████████      19%
Fare         ████████████          14%
Age          ██████████            12%
Pclass       ████████              10%
```

Two of the top five (`Title`, engineered from the Name column) were features that didn't exist in the original dataset at all.

---

## What I Learned

Data cleaning and feature engineering take up **80% of the work**. The actual model training is just a few lines of code. The real skill is in understanding your data deeply before touching any algorithm.

Specific lessons:
- Never ignore missingness — even the absence of data (`HasCabin`) carries signal
- Encoding mistakes (ordinal vs one-hot) can silently hurt your model
- Always evaluate with precision, recall, and F1 — accuracy alone misleads
- Cross-validation gives far more reliable results than a single 80/20 split
- Feature importance scores are the model's way of confirming your EDA was right

This is just the beginning 🚀

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib + Seaborn | EDA visualisations |
| Scikit-learn | Preprocessing, models, evaluation |
| Jupyter Notebook | Development environment |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook titanic_survival.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset) and place `Titanic-Dataset.csv` in the root folder before running.

---

## Project Structure

```
titanic-survival-prediction/
│
├── Titanic-Dataset.csv          # Raw dataset (download from Kaggle)
├── titanic_survival.ipynb       # Main notebook — full pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Author

**Zishan** — IIT Madras  
[LinkedIn](https://www.linkedin.com/in/zishan-ahmad-155b24239/) · [GitHub](https://github.com/zishaniitm)

---

*If you found this useful, drop a ⭐ on the repo!*
