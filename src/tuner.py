from .evals.metrics import RegressionMetrics
from .logger.logger import Logger


class ModelTuner:
    def __init__(self, model_class, param_grid, model_name="model"):
        self.model_class = model_class
        self.param_grid = param_grid
        self.model_name = model_name
        self.logger = Logger()

    def tune(self, X_train, X_test, y_train, y_test):
        best_result = None
        best_params = None
        best_model = None

        for params in self.param_grid:
            model = self.model_class(**params)  # ✔ fresh model

            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = RegressionMetrics(y_test, y_pred)
            result = metrics.get_all_metrics()

            log_entry = {
                "model": self.model_name,
                **params,
                "RMSE": result["RMSE"],
                "MAE": result["MAE"],
                "R2": result["R2"]
            }

            self.logger.logs.append(log_entry)

            # update best
            if best_result is None or result["RMSE"] < best_result["RMSE"]:
                best_result = result
                best_params = params
                best_model = model

        self.logger.save()

        return best_model, best_params, best_result