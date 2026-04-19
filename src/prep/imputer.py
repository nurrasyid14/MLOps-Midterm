import pandas as pd


class DataImputer:
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []

    def fit(self, df: pd.DataFrame):
        self.numeric_cols = df.select_dtypes(include=["number"]).columns
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        # numerik
        self.means = df[self.numeric_cols].mean()

        # handle kalau tidak ada kolom kategorik
        if len(self.categorical_cols) > 0:
            self.modes = df[self.categorical_cols].mode().iloc[0]
        else:
            self.modes = pd.Series(dtype=object)

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # numerik
        df[self.numeric_cols] = df[self.numeric_cols].fillna(self.means)

        # kategorik (kalau ada)
        if len(self.categorical_cols) > 0:
            df[self.categorical_cols] = df[self.categorical_cols].fillna(self.modes)

        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)