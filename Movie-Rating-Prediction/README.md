# 🎬 Movie Rating Prediction — IMDb India

> Predicting IMDb movie ratings using machine learning on Indian cinema data.  
> Built with Python · scikit-learn · pandas · matplotlib · seaborn

---

## 📌 Project Overview

This project builds a machine learning regression model that predicts the **IMDb rating of Indian movies** based on features like genre, director reputation, actor popularity, votes, and release year.

The goal is to uncover what actually drives a movie's rating — and build a model accurate enough to estimate it within **0.41 rating points on average**.

---

## 📊 Dataset

- **Source**: [IMDb India Movies — Kaggle](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)
- **Original size**: 15,509 movies × 10 columns
- **After cleaning**: 7,893 usable records
- **Features**: Name, Year, Duration, Genre, Rating, Votes, Director, Actor 1, Actor 2, Actor 3

---

## 🔁 Project Pipeline

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Insights
```

### Phase 1 — Exploratory Data Analysis (EDA)
- Inspected data shape, types, and distributions
- Identified missing values per column
- Visualized rating distribution, genre frequency, and top directors/actors
- Found average rating of **5.84** with slight left skew

### Phase 2 — Data Preprocessing
- Dropped rows with missing `Rating` (target column — cannot be imputed)
- Extracted 4-digit year from strings like `"2019 (I)"` using regex
- Extracted numeric duration from strings like `"120 min"`
- Removed commas from `Votes` column and converted to numeric
- Filled missing categorical columns with `"Unknown"`
- Removed outliers using the **IQR method** (kept ratings between 2.05 and 9.65)

### Phase 3 — Feature Engineering
| Feature | Technique | Reason |
|---|---|---|
| `Primary_Genre` | Split first genre from list | Reduce dimensionality |
| `Genre_Encoded` | Label Encoding | Convert text to integer |
| `Director_Score` | Mean Encoding (avg rating) | Encode directorial reputation |
| `Actor1/2/3_Score` | Mean Encoding (avg rating) | Encode actor popularity |
| `Votes_Log` | Log Transform `log1p(x)` | Normalize heavily skewed distribution |

### Phase 4 — Model Training
Three regression models trained and compared:
- **Linear Regression** — simple interpretable baseline
- **Random Forest** — ensemble of 100 decision trees
- **Gradient Boosting** — sequential error-correcting trees

### Phase 5 — Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross validation on Random Forest
- Tuned: `n_estimators`, `max_depth`, `min_samples_split`

---

## 📈 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.4931 | 0.6681 | 0.7642 |
| Random Forest (baseline) | 0.4185 | 0.6141 | 0.8008 |
| Gradient Boosting | 0.4394 | 0.6130 | 0.8015 |
| **Random Forest (tuned)** | **0.4153** | **0.6104** | **0.8032** |

> ✅ Best Model: **Tuned Random Forest**  
> ✅ The model predicts movie ratings within **0.41 points** on average

---

## 🔍 Key Insights

```
Director_Score    ████████████████████████████████████  61.9%
Actor3_Score      ██████                                11.5%
Actor2_Score      █████                                  9.8%
Actor1_Score      ████                                   7.5%
Votes_Log         ██                                     4.5%
Year              █                                      2.4%
Duration          █                                      1.6%
Genre_Encoded     ░                                      0.9%
```

**Finding 1:** Director reputation alone accounts for **62% of the model's predictive power** — the single most important factor in Indian cinema ratings.

**Finding 2:** Supporting actors (Actor 2 & 3) carry more weight than the lead actor — strong ensemble casts signal quality productions.

**Finding 3:** Genre contributes less than 1% — a talented director can make any genre work.

---

## 🗂️ Project Structure

```
movie-rating-prediction/
│
├── IMDb Movies India.csv          # Raw dataset
├── movie_rating_prediction.ipynb  # Main Colab notebook
├── movie_rating_model.pkl         # Saved trained model
├── scaler.pkl                     # Saved StandardScaler
└── README.md                      # Project documentation
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/movie-rating-prediction.git
cd movie-rating-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

**3. Open in Google Colab**  
Upload `movie_rating_prediction.ipynb` to [colab.research.google.com](https://colab.research.google.com) and run all cells top to bottom.

**4. Load saved model for predictions**
```python
import joblib
import numpy as np
import pandas as pd

model  = joblib.load('movie_rating_model.pkl')
scaler = joblib.load('scaler.pkl')

movie = pd.DataFrame([{
    'Year': 2020, 'Duration': 145, 'Genre_Encoded': 1,
    'Director_Score': 7.8, 'Actor1_Score': 7.2,
    'Actor2_Score': 6.8, 'Actor3_Score': 6.5,
    'Votes_Log': np.log1p(100000)
}])

predicted_rating = model.predict(scaler.transform(movie))
print(f"Predicted Rating: {predicted_rating[0]:.2f} / 10")
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-darkblue?style=flat-square&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![Colab](https://img.shields.io/badge/Google%20Colab-notebook-yellow?style=flat-square&logo=googlecolab)

---

## 👤 Author

**Zishan**  
Intern @ [Codsoft]  
[LinkedIn](https://www.linkedin.com/in/zishan-ahmad-155b24239/) · [GitHub](https://github.com/zishaniitm)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
---

*If you found this useful, drop a ⭐ on the repo!*
