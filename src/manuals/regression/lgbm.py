# src/manuals/regression_lgbm.py

from lightgbm import LGBMRegressor


class LGBMModel:
    def __init__(self, **params):
        self.model = LGBMRegressor(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)