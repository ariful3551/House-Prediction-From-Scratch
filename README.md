# House-Prediction-From-Scratch

A hands-on machine learning project that predicts California house prices. The goal is to learn the full workflow end to end: analyze the data with pandas, build the core algorithms from scratch with NumPy, then re-implement and compare them with scikit-learn.

## Learning approach

1. **Explore** the data with pandas, matplotlib and seaborn.
2. **Clean** the data and engineer features.
3. **Split and scale** using my own NumPy code (no leakage from the test set).
4. **Train models from scratch** with NumPy: linear regression (normal equation and gradient descent) and ridge regression.
5. **Re-implement with scikit-learn** and compare the results against my NumPy versions.
6. **Learn new tools step by step**: pipelines, cross-validation, tree-based models, tuning.

## Dataset

California Housing data: 20,640 rows, 9 numeric features and 1 categorical feature (`ocean_proximity`). The target is `median_house_value`.

## Key findings so far

- Only `total_bedrooms` has missing values (207 rows, about 1%).
- `median_house_value` (max 500,001) and `housing_median_age` (max 52) look capped at their maximum.
- `total_rooms`, `total_bedrooms`, `population` and `households` have extreme maximum values compared with their 75th percentile.

## Project structure

```
House-Prediction-From-Scratch/
├── data/
│   └── raw/            # original housing.csv
├── notebooks/          # numbered analysis and modelling notebooks
├── src/                # reusable Python code
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Then open the notebooks in `notebooks/` in order.

## Roadmap

- [x] Project setup
- [x] 01 Data overview (shape, dtypes, missing values, summary statistics)
- [ ] 02 Distributions and skewness
- [ ] 03 Capping and outliers
- [ ] 04 Geographic analysis
- [ ] 05 Correlation and key findings
- [ ] 06 Cleaning and feature engineering
- [ ] 07 Train/validation/test split and scaling in NumPy
- [ ] 08 Linear regression from scratch (normal equation, gradient descent)
- [ ] 09 Ridge regression and evaluation (MAE, RMSE, R²)
- [ ] 10 scikit-learn vs NumPy comparison
- [ ] 11 Cross-validation and tuning
- [ ] 12 Error analysis and limitations

## Results

Results will be added here once the models are trained.