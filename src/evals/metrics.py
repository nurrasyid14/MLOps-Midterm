import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def rmse(self):
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def get_all_metrics(self):
        return {
            "RMSE": self.rmse(),
            "MAE": self.mae(),
            "R2": self.r2()
        }