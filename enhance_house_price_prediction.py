import numpy as np
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('housing.csv')
df.head()

X_boston = df.drop(columns=['median_house_value','ocean_proximity'])
Y_boston = df['median_house_value']

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_boston, Y_boston , train_size=0.80, test_size=0.20, random_state=123)
print('Train/Test Sets Sizes : ',X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

np.round(X_train.describe(), 1)

np.round(Y_train.describe(), 1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_scaled


from sklearn.impute import KNNImputer,SimpleImputer
knn = KNNImputer(n_neighbors=3,weights='distance')

X_train_trf = knn.fit_transform(X_train_scaled)
X_test_trf = knn.transform(X_test_scaled)

lr = LinearRegression()
dt = DecisionTreeRegressor()
knn = KNeighborsRegressor()


lr.fit(X_train_trf,Y_train)
dt.fit(X_train_trf,Y_train)
knn.fit(X_train_trf,Y_train)

y_pred1 = lr.predict(X_test_trf)
y_pred2 = dt.predict(X_test_trf)
y_pred3 = knn.predict(X_test_trf)


print("R^2 score for LR",r2_score(Y_test,y_pred1))
print("R^2 score for DT",r2_score(Y_test,y_pred2))
print("R^2 score for KNN",r2_score(Y_test,y_pred3))

from sklearn.ensemble import BaggingRegressor

bag_regressor = BaggingRegressor(random_state=1)
bag_regressor.fit(X_train_trf, Y_train)
Y_preds = bag_regressor.predict(X_test_trf)

print('Training Coefficient of R^2 : %.3f'%bag_regressor.score(X_train_trf, Y_train))
print('Test Coefficient of R^2 : %.3f'%bag_regressor.score(X_test_trf, Y_test))
n_samples = X_train_trf.shape[0]
n_features = X_train_trf.shape[1]

print(f"Number of samples: {n_samples}")
print(f"Number of features: {n_features}")

%%time

n_samples = X_train_trf.shape[0]
n_features = X_train_trf.shape[1]

params = {'base_estimator': [None, LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor()],
          'n_estimators': [20,50,100],
          'max_samples': [0.5,1.0],
          'max_features': [0.5,1.0],
          'bootstrap': [True, False],
          'bootstrap_features': [True, False]}

bagging_regressor_grid = GridSearchCV(BaggingRegressor(random_state=1, n_jobs=-1), param_grid =params, cv=3, n_jobs=-1, verbose=1)
bagging_regressor_grid.fit(X_train_trf, Y_train)

print('Train R^2 Score : %.3f'%bagging_regressor_grid.best_estimator_.score(X_train_trf, Y_train))
print('Test R^2 Score : %.3f'%bagging_regressor_grid.best_estimator_.score(X_test_trf, Y_test))
print('Best R^2 Score Through Grid Search : %.3f'%bagging_regressor_grid.best_score_)
print('Best Parameters : ',bagging_regressor_grid.best_params_)

from sklearn.ensemble import BaggingRegressor
bag_regressor = BaggingRegressor(
    random_state=1,
    n_jobs=-1,
    base_estimator=None,
    bootstrap=True,
    bootstrap_features=False,
    max_features=1.0,
    max_samples=1.0,
    n_estimators=100
)

bag_regressor.fit(X_train_trf, Y_train)

Y_preds = bag_regressor.predict(X_test_trf)

print('Training Coefficient of R^2 : %.3f' % bag_regressor.score(X_train_trf, Y_train))
print('Test Coefficient of R^2 : %.3f' % bag_regressor.score(X_test_trf, Y_test))
