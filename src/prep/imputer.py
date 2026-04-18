import pandas as pd


class DataImputer:
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []

    def fit(self, df: pd.DataFrame):
        self.numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        self.means = df[self.numeric_cols].mean()
        self.modes = df[self.categorical_cols].mode().iloc[0]

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # isi numerik dengan mean
        df[self.numeric_cols] = df[self.numeric_cols].fillna(self.means)

        # isi kategorik dengan modus
        df[self.categorical_cols] = df[self.categorical_cols].fillna(self.modes)

        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)