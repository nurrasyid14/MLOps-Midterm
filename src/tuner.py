from .evals.metrics import RegressionMetrics
from .logger.logger import Logger


class RidgeTuner:
    def __init__(self, model, alphas):
        self.model = model
        self.alphas = alphas
        self.logger = Logger()

    def tune(self, X_train, X_test, y_train, y_test):
        best_result = None
        best_alpha = None

        for alpha in self.alphas:
            self.model.set_alpha(alpha)
            self.model.train(X_train, y_train)

            y_pred = self.model.predict(X_test)

            metrics = RegressionMetrics(y_test, y_pred)
            result = metrics.get_all_metrics()

            self.logger.log(alpha, result)

            if best_result is None or result["RMSE"] < best_result["RMSE"]:
                best_result = result
                best_alpha = alpha

        self.logger.save()

        return best_alpha, best_result