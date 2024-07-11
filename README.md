# Enhanced_house_price_prediction
This project focuses on predicting housing prices using a dataset with various property features. It includes data preprocessing (scaling, imputation), model training (Linear Regression, Decision Tree, K-Nearest Neighbors), and ensemble methods (BaggingRegressor) with hyperparameter tuning to optimize predictions and evaluate model performance.
## Overview
<h3>Data Preparation</h3>: Handles missing values, scales features, and splits the dataset into training and testing sets.

<h3>Model Development</h3>: Trains and evaluates Linear Regression, Decision Tree, and K-Nearest Neighbors models.

<h3>Ensemble Optimization</h3>: Applies BaggingRegressor with GridSearchCV for hyperparameter tuning to improve prediction accuracy.

## Files
- `housing.csv`: The source dataset used for the analysis.

## Preview
<pre>
The code demonstrates a comprehensive approach to housing price prediction, starting from data preprocessing (missing value imputation and feature scaling) to model training and evaluation. It includes various regression models (Linear Regression, Decision Tree, K-Nearest Neighbors) and leverages ensemble learning with BaggingRegressor. Hyperparameter optimization is performed using GridSearchCV to fine-tune the ensemble model for better accuracy. The project aims to enhance predictive performance and assess different models' effectiveness in forecasting housing prices.

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

## Data Sources
- The Kaggle provided the housing' data.
