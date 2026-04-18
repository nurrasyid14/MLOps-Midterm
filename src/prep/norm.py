from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class MinMaxNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.columns = None

    def fit(self, X: pd.DataFrame):
        self.columns = X.columns
        self.scaler.fit(X)

    def transform(self, X: pd.DataFrame):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)