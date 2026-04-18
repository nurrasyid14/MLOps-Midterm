import pandas as pd


class CorrelationAnalyzer:
    def __init__(self):
        self.corr_matrix = None

    def fit(self, df: pd.DataFrame):
        self.corr_matrix = df.corr(numeric_only=True)
        return self.corr_matrix

    def get_high_correlation(self, threshold=0.8):
        corr_pairs = []
        corr = self.corr_matrix

        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > threshold:
                    corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

        return corr_pairs

    def drop_highly_correlated(self, df: pd.DataFrame, threshold=0.8):
        corr = df.corr(numeric_only=True).abs()
        upper = corr.where(~(corr == 1.0)).where(
            pd.np.triu(pd.np.ones(corr.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop), to_drop