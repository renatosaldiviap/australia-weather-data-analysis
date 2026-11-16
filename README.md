# 🌧️ Rainfall Prediction in Australia | End-to-End ML Pipeline (CRISP-DM)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-2E7D32?style=for-the-badge)

## 📌 Project Overview

This project builds an end-to-end Machine Learning classification pipeline designed to predict next-day rainfall (`RainTomorrow`) across Australian meteorological stations using **142,193 cleaned records**, strictly adhering to the **CRISP-DM** industry methodology.

Using **Python**, **Scikit-Learn**, and **LightGBM**, raw weather observations were audited for data leakage, preprocessed through a unified `ColumnTransformer`, and benchmarked under class imbalance (~78% dry days vs. ~22% rainy days) to maximize minority-class recall and ROC-AUC.

## 🛠️ Tech Stack & Pipeline Architecture

- **Python (`Pandas`, `NumPy`):** Structural auditing, missing-value filtering (>35% threshold), and data leakage removal (`RISK_MM`).
- **Preprocessing & Modeling (`Scikit-Learn`, `LightGBM`):** Leakage-free `ColumnTransformer` (median/mode imputation, `StandardScaler`, `OneHotEncoder`) paired with class-balanced gradient boosting.
- **Serialization & Inference (`Joblib`):** End-to-end unified pipeline serialized into `weather_rain_prediction_lgbm_pipeline.joblib` for production-ready inference.

```text
[ Weather Data (142k rows) ] ──► [ Leakage & Null Audit ] ──► [ ColumnTransformer Pipeline ] ──► [ LightGBM Champion Model ]
```

## 🔄 CRISP-DM Lifecycle & Code Highlights

1. **Business & Data Understanding:** Prioritized ROC-AUC and Rain Recall over raw accuracy to handle class imbalance (22.42% positive class). Detected and dropped `RISK_MM` to prevent target leakage, along with high-nullity features (`Sunshine`, `Evaporation`, `Cloud3pm`, `Cloud9am`).
2. **Data Preparation:** Executed an 80/20 stratified train/test split and encapsulated numeric and categorical transformations inside a unified pipeline (110 post-encoded features).
3. **Deployment & Inference:** Serialized the trained pipeline and built `predict_rain_tomorrow()` to output probability-based rainfall risk levels on raw input records.

```python
# Unified leakage-free preprocessing and LightGBM classification pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])),
    ('classifier', LGBMClassifier(scale_pos_weight=3.46, random_state=42))
])
```

## 📊 Model Benchmarking & Evaluation

Three algorithmic architectures were evaluated on the stratified test set using class-weight balancing:

| Model Architecture | ROC-AUC | Recall (Rain) | Precision (Rain) | F1-Score | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8637 | 0.77 | 0.52 | 0.62 | 0.79 |
| Random Forest | 0.8625 | 0.74 | 0.54 | 0.62 | 0.80 |
| **LightGBM (Champion)** | **0.8843** | **0.79** | **0.55** | **0.64** | **0.81** |

## 💡 Key Analytical & Meteorological Insights

1. **Strong Minority-Class Detection:** The champion LightGBM model achieved **0.8843 ROC-AUC** and captured **78.65%** of actual rainy days (5,014 true positives), limiting false negatives to 1,361 instances.
2. **Physical Feature Alignment:** Late-day atmospheric pressure (`Pressure3pm`), afternoon humidity (`Humidity3pm`), and wind gust speed (`WindGustSpeed`) emerged as the top predictors, aligning with real meteorological dynamics.
3. **Leakage Prevention Impact:** Eliminating `RISK_MM` prevented artificially inflated metrics and ensured the model relies solely on true prior-day weather signals.

## 📁 Repository Structure

```text
├── weather_rainfall_prediction_crispdm.ipynb   # Full CRISP-DM notebook & evaluation
├── .gitignore                                  # Ignores raw datasets and binary models
└── README.md                                   # Technical documentation
```

## 🚀 How to Run

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/renatosaldiviap/australia-weather-data-analysis.git
   ```
2. Download the dataset (`weatherAUS.csv`) from [Kaggle - Australia Weather Data](https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data) and place it in the project root directory.
3. Open `weather_rainfall_prediction_crispdm.ipynb` in Google Colab or Jupyter Notebook and run all cells sequentially.
