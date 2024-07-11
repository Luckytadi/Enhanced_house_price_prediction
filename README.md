# Enhanced_house_price_prediction
## Overview
<b>Data Preparation</b>: Handles missing values, scales features, and splits the dataset into training and testing sets.

<b>Model Development</b>: Trains and evaluates Linear Regression, Decision Tree, and K-Nearest Neighbors models.

<b>Ensemble Optimization</b>: Applies BaggingRegressor with GridSearchCV for hyperparameter tuning to improve prediction accuracy.

## Files
- `housing.csv`: The source dataset used for the analysis.

## Preview
<pre>
fitting 3 folds for each of 192 candidates, totalling 576 fits
Train R^2 Score : 0.975
Test R^2 Score : 0.829
Best R^2 Score Through Grid Search : 0.806
Best Parameters :  {'base_estimator': None, 'bootstrap': True, 'bootstrap_features': False, 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 100}
CPU times: user 18.9 s, sys: 4.82 s, total: 23.7 s
Wall time: 28min 21s

Training Coefficient of R^2 : 0.975
Test Coefficient of R^2 : 0.829
</pre>

## How to Use
1. Clone this repository.
2. Open `enhanced_house_price_prediction.py' with Jupyter Notebook.

## Data Source
- The Kaggle provided the housing' data.
