from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV

import numpy as np

class using_RandomForestRegressor():
    def __init__(self):
        pass

    def normal_split_RandomForestRegressor(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def GridSearchCV_RandomForestRegressor(X,y):
        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestRegressor(random_state=42)
        param_search = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30]
        }
        gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search)
        gsearch.fit(X, y)
        best_model = gsearch.best_estimator_

        return best_model
    
class using_LinearRidge():
    def __init__(self):
        pass
    
    def normal_split_LinearRidge(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(random_state=42)
        model.fit(X_train, y_train)
        return model

    def GridSearchCV_LinearRidge(X,y):
        tscv = TimeSeriesSplit(n_splits=5)

        model = Ridge(random_state=42)

        param_Search ={
            'alpha': np.logspace(-3, 3, 7) # 10^-3 até 10^3
        }

        gsearch = GridSearchCV(estimator=model, param_grid=param_Search, cv=tscv, scoring='neg_mean_squared_error')

        gsearch.fit(X,y)

        best_model = gsearch.best_estimator_

        return best_model